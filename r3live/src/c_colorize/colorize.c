#include "colorize.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static double c_colorize_clamp(double value, double min_value, double max_value)
{
    if (value < min_value)
    {
        return min_value;
    }
    if (value > max_value)
    {
        return max_value;
    }
    return value;
}

static void c_colorize_transform_point(const c_colorize_camera_t *camera,
                                       const double world_point[3],
                                       double camera_point[3])
{
    camera_point[0] = camera->rotation[0] * world_point[0] + camera->rotation[1] * world_point[1] +
                      camera->rotation[2] * world_point[2] + camera->translation[0];
    camera_point[1] = camera->rotation[3] * world_point[0] + camera->rotation[4] * world_point[1] +
                      camera->rotation[5] * world_point[2] + camera->translation[1];
    camera_point[2] = camera->rotation[6] * world_point[0] + camera->rotation[7] * world_point[1] +
                      camera->rotation[8] * world_point[2] + camera->translation[2];
}

static int c_colorize_point_in_bounds(const c_colorize_camera_t *camera, double u, double v)
{
    const double margin = camera->fov_margin;
    const double min_u = margin * (double)camera->image_cols + 1.0;
    const double min_v = margin * (double)camera->image_rows + 1.0;
    const double max_u = (1.0 - margin) * (double)camera->image_cols;
    const double max_v = (1.0 - margin) * (double)camera->image_rows;

    if (u >= min_u && ceil(u) < max_u && v >= min_v && ceil(v) < max_v)
    {
        return 1;
    }
    return 0;
}

static int c_colorize_sample_nearest(const c_colorize_camera_t *camera,
                                     const c_colorize_image_t *image,
                                     double u,
                                     double v,
                                     unsigned char bgr[3])
{
    int col = (int)u;
    int row = (int)v;
    const unsigned char *pixel = NULL;

    if (camera->image_cols <= 0 || camera->image_rows <= 0 || image->channels < 3)
    {
        return 0;
    }

    col = (int)c_colorize_clamp((double)col, 0.0, (double)(camera->image_cols - 1));
    row = (int)c_colorize_clamp((double)row, 0.0, (double)(camera->image_rows - 1));
    pixel = image->data + row * image->row_stride + col * image->channels;
    bgr[0] = pixel[0];
    bgr[1] = pixel[1];
    bgr[2] = pixel[2];
    return 1;
}

static double c_colorize_sample_bilinear_channel(const c_colorize_image_t *image,
                                                 int channels,
                                                 double u,
                                                 double v,
                                                 int channel)
{
    int floor_row = (int)floor(v);
    int floor_col = (int)floor(u);
    int ceil_row = floor_row + 1;
    int ceil_col = floor_col + 1;
    double frac_row = v - (double)floor_row;
    double frac_col = u - (double)floor_col;
    const unsigned char *p00 = image->data + floor_row * image->row_stride + floor_col * channels;
    const unsigned char *p10 = image->data + ceil_row * image->row_stride + floor_col * channels;
    const unsigned char *p01 = image->data + floor_row * image->row_stride + ceil_col * channels;
    const unsigned char *p11 = image->data + ceil_row * image->row_stride + ceil_col * channels;

    return (1.0 - frac_row) * (1.0 - frac_col) * (double)p00[channel] +
           frac_row * (1.0 - frac_col) * (double)p10[channel] +
           (1.0 - frac_row) * frac_col * (double)p01[channel] +
           frac_row * frac_col * (double)p11[channel];
}

static int c_colorize_sample_bilinear(const c_colorize_camera_t *camera,
                                      const c_colorize_image_t *image,
                                      double u,
                                      double v,
                                      unsigned char bgr[3])
{
    int channel = 0;
    double max_u = (double)(camera->image_cols - 2);
    double max_v = (double)(camera->image_rows - 2);

    if (camera->image_cols < 2 || camera->image_rows < 2 || image->channels < 3)
    {
        return c_colorize_sample_nearest(camera, image, u, v, bgr);
    }

    u = c_colorize_clamp(u, 0.0, max_u);
    v = c_colorize_clamp(v, 0.0, max_v);
    for (channel = 0; channel < 3; ++channel)
    {
        double sampled = c_colorize_sample_bilinear_channel(image, image->channels, u, v, channel);
        bgr[channel] = (unsigned char)(sampled + 0.5);
    }
    return 1;
}

static int c_colorize_sample_pixel(const c_colorize_camera_t *camera,
                                   const c_colorize_image_t *image,
                                   double u,
                                   double v,
                                   int sample_mode,
                                   unsigned char bgr[3])
{
    if (sample_mode == C_COLORIZE_SAMPLE_NEAREST)
    {
        return c_colorize_sample_nearest(camera, image, u, v, bgr);
    }
    return c_colorize_sample_bilinear(camera, image, u, v, bgr);
}

int c_colorize_select_point(const c_colorize_camera_t *cameras,
                            const c_colorize_image_t *images,
                            int camera_count,
                            const double world_point[3],
                            int sample_mode,
                            int selection_mode,
                            c_colorize_result_t *result)
{
    int camera_idx = 0;
    int saw_out_of_bounds = 0;
    c_colorize_result_t best_result;
    double best_distance = DBL_MAX;

    if (result == NULL)
    {
        return 0;
    }

    memset(result, 0, sizeof(*result));
    result->camera_index = -1;
    result->failure_reason = C_COLORIZE_FAIL_INVALID_INPUT;

    if (cameras == NULL || images == NULL || world_point == NULL || camera_count <= 0)
    {
        return 0;
    }

    memset(&best_result, 0, sizeof(best_result));
    best_result.camera_index = -1;
    best_result.failure_reason = C_COLORIZE_FAIL_INVALID_INPUT;

    for (camera_idx = 0; camera_idx < camera_count; ++camera_idx)
    {
        double camera_point[3];
        double u = 0.0;
        double v = 0.0;
        double distance = 0.0;
        unsigned char sampled_bgr[3];
        c_colorize_result_t current_result;

        if (images[camera_idx].data == NULL || images[camera_idx].channels < 3 || images[camera_idx].row_stride <= 0 ||
            cameras[camera_idx].image_cols <= 0 || cameras[camera_idx].image_rows <= 0)
        {
            continue;
        }

        c_colorize_transform_point(&cameras[camera_idx], world_point, camera_point);
        if (camera_point[2] < 0.001)
        {
            continue;
        }

        u = camera_point[0] * cameras[camera_idx].fx / camera_point[2] + cameras[camera_idx].cx;
        v = camera_point[1] * cameras[camera_idx].fy / camera_point[2] + cameras[camera_idx].cy;
        if (!c_colorize_point_in_bounds(&cameras[camera_idx], u, v))
        {
            saw_out_of_bounds = 1;
            continue;
        }

        if (!c_colorize_sample_pixel(&cameras[camera_idx], &images[camera_idx], u, v, sample_mode, sampled_bgr))
        {
            continue;
        }

        memset(&current_result, 0, sizeof(current_result));
        current_result.success = 1;
        current_result.camera_index = camera_idx;
        current_result.failure_reason = C_COLORIZE_FAIL_NONE;
        current_result.u = u;
        current_result.v = v;
        current_result.camera_distance =
            sqrt(camera_point[0] * camera_point[0] + camera_point[1] * camera_point[1] + camera_point[2] * camera_point[2]);
        current_result.bgr[0] = sampled_bgr[0];
        current_result.bgr[1] = sampled_bgr[1];
        current_result.bgr[2] = sampled_bgr[2];

        if (selection_mode == C_COLORIZE_SELECT_FIRST_VALID)
        {
            *result = current_result;
            return 1;
        }

        distance = current_result.camera_distance;
        if (distance < best_distance)
        {
            best_distance = distance;
            best_result = current_result;
        }
    }

    if (best_result.success)
    {
        *result = best_result;
        return 1;
    }

    result->success = 0;
    result->camera_index = -1;
    result->failure_reason = saw_out_of_bounds ? C_COLORIZE_FAIL_OUT_OF_BOUNDS : C_COLORIZE_FAIL_BEHIND_CAMERA;
    return 0;
}
