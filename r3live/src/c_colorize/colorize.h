#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    C_COLORIZE_FAIL_NONE = 0,
    C_COLORIZE_FAIL_BEHIND_CAMERA = 1,
    C_COLORIZE_FAIL_OUT_OF_BOUNDS = 2,
    C_COLORIZE_FAIL_INVALID_INPUT = 3
};

enum
{
    C_COLORIZE_SAMPLE_NEAREST = 0,
    C_COLORIZE_SAMPLE_BILINEAR = 1
};

enum
{
    C_COLORIZE_SELECT_FIRST_VALID = 0,
    C_COLORIZE_SELECT_NEAREST_DISTANCE = 1
};

typedef struct
{
    double fx;
    double fy;
    double cx;
    double cy;
    double rotation[9];
    double translation[3];
    int image_rows;
    int image_cols;
    double fov_margin;
} c_colorize_camera_t;

typedef struct
{
    const unsigned char *data;
    int row_stride;
    int channels;
} c_colorize_image_t;

typedef struct
{
    int success;
    int camera_index;
    int failure_reason;
    double u;
    double v;
    double camera_distance;
    unsigned char bgr[3];
} c_colorize_result_t;

int c_colorize_select_point(const c_colorize_camera_t *cameras,
                            const c_colorize_image_t *images,
                            int camera_count,
                            const double world_point[3],
                            int sample_mode,
                            int selection_mode,
                            c_colorize_result_t *result);

#ifdef __cplusplus
}
#endif
