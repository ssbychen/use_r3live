/* 
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored, 
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored, 
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package." 
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping." 
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry 
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for 
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision 
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "pointcloud_rgbd.hpp"
#include "../optical_flow/lkpyramid.hpp"
#include "../c_colorize/colorize.h"
#include <sstream>
extern Common_tools::Cost_time_logger g_cost_time_logger;
extern std::shared_ptr<Common_tools::ThreadPool> m_thread_pool_ptr;
cv::RNG g_rng = cv::RNG(0);
// std::atomic<long> g_pts_index(0);

#ifndef R3LIVE_USE_C_COLORIZE
#define R3LIVE_USE_C_COLORIZE 1
#endif

#ifndef R3LIVE_VERIFY_C_COLORIZE
#define R3LIVE_VERIFY_C_COLORIZE 0
#endif

namespace
{
enum class ProjectionFailureReason
{
    kNone = C_COLORIZE_FAIL_NONE,
    kBehindCamera = C_COLORIZE_FAIL_BEHIND_CAMERA,
    kOutOfBounds = C_COLORIZE_FAIL_OUT_OF_BOUNDS,
    kInvalidInput = C_COLORIZE_FAIL_INVALID_INPUT
};

struct ColorizeExecutionResult
{
    bool                    success = false;
    ProjectionFailureReason failure_reason = ProjectionFailureReason::kInvalidInput;
    int                     camera_index = -1;
    double                  u = 0.0;
    double                  v = 0.0;
    double                  camera_distance = 0.0;
    vec_3                   rgb = vec_3::Zero();
};

struct ColorizeDebugStats
{
    long              projection_success = 0;
    long              projection_fail = 0;
    long              behind_camera = 0;
    long              out_of_bounds = 0;
    long              invalid_input = 0;
    long              legacy_projection_disagreement = 0;
    long              legacy_color_diff_samples = 0;
    double            legacy_color_diff_sum = 0.0;
    double            legacy_color_diff_max = 0.0;
    std::vector<long> per_camera_hits;

    explicit ColorizeDebugStats( size_t camera_count = 1 ) : per_camera_hits( camera_count, 0 ) {}

    void record_result( const ColorizeExecutionResult &result )
    {
        if ( result.success )
        {
            projection_success++;
            if ( result.camera_index >= 0 )
            {
                if ( per_camera_hits.size() <= static_cast<size_t>( result.camera_index ) )
                {
                    per_camera_hits.resize( result.camera_index + 1, 0 );
                }
                per_camera_hits[ result.camera_index ]++;
            }
            return;
        }

        projection_fail++;
        switch ( result.failure_reason )
        {
        case ProjectionFailureReason::kBehindCamera:
            behind_camera++;
            break;
        case ProjectionFailureReason::kOutOfBounds:
            out_of_bounds++;
            break;
        default:
            invalid_input++;
            break;
        }
    }

    void merge( const ColorizeDebugStats &other )
    {
        projection_success += other.projection_success;
        projection_fail += other.projection_fail;
        behind_camera += other.behind_camera;
        out_of_bounds += other.out_of_bounds;
        invalid_input += other.invalid_input;
        legacy_projection_disagreement += other.legacy_projection_disagreement;
        legacy_color_diff_samples += other.legacy_color_diff_samples;
        legacy_color_diff_sum += other.legacy_color_diff_sum;
        legacy_color_diff_max = std::max( legacy_color_diff_max, other.legacy_color_diff_max );
        if ( per_camera_hits.size() < other.per_camera_hits.size() )
        {
            per_camera_hits.resize( other.per_camera_hits.size(), 0 );
        }
        for ( size_t idx = 0; idx < other.per_camera_hits.size(); ++idx )
        {
            per_camera_hits[ idx ] += other.per_camera_hits[ idx ];
        }
    }
};

struct ThreadRenderReport
{
    double             cost_time = 0.0;
    long               render_updates = 0;
    ColorizeDebugStats stats;

    explicit ThreadRenderReport( size_t camera_count = 1 ) : stats( camera_count ) {}
};

inline pcl::PointXYZI make_pcl_point( const vec_3 &pt_w )
{
    pcl::PointXYZI pt;
    pt.x = pt_w( 0 );
    pt.y = pt_w( 1 );
    pt.z = pt_w( 2 );
    return pt;
}

ColorizeExecutionResult run_legacy_colorize( const std::shared_ptr< Image_frame > &img_ptr, const vec_3 &pt_w )
{
    ColorizeExecutionResult result;
    if ( img_ptr == nullptr || img_ptr->m_img.empty() )
    {
        return result;
    }

    pcl::PointXYZI pcl_pt = make_pcl_point( pt_w );
    if ( img_ptr->project_3d_to_2d( pcl_pt, img_ptr->m_cam_K, result.u, result.v, 1.0 ) == false )
    {
        result.failure_reason = ProjectionFailureReason::kBehindCamera;
        return result;
    }
    if ( img_ptr->if_2d_points_available( result.u, result.v, 1.0 ) == false )
    {
        result.failure_reason = ProjectionFailureReason::kOutOfBounds;
        return result;
    }

    result.success = true;
    result.failure_reason = ProjectionFailureReason::kNone;
    result.camera_index = 0;
    result.camera_distance = ( pt_w - img_ptr->m_pose_w2c_t ).norm();
    result.rgb = img_ptr->get_rgb( result.u, result.v, 0 );
    return result;
}

ColorizeExecutionResult run_c_colorize( const std::shared_ptr< Image_frame > &img_ptr, const vec_3 &pt_w )
{
    ColorizeExecutionResult result;
    if ( img_ptr == nullptr || img_ptr->m_img.empty() || img_ptr->m_img.channels() < 3 )
    {
        return result;
    }

    c_colorize_camera_t camera = {};
    c_colorize_image_t  image = {};
    c_colorize_result_t c_result = {};
    Eigen::Matrix3d     rotation = img_ptr->m_pose_c2w_q.toRotationMatrix();
    const int           image_rows = img_ptr->m_img_rows > 0 ? img_ptr->m_img_rows : img_ptr->m_img.rows;
    const int           image_cols = img_ptr->m_img_cols > 0 ? img_ptr->m_img_cols : img_ptr->m_img.cols;
    const double        world_point[ 3 ] = { pt_w( 0 ), pt_w( 1 ), pt_w( 2 ) };

    camera.fx = img_ptr->fx;
    camera.fy = img_ptr->fy;
    camera.cx = img_ptr->cx;
    camera.cy = img_ptr->cy;
    camera.image_rows = image_rows;
    camera.image_cols = image_cols;
    camera.fov_margin = img_ptr->m_fov_margin;
    for ( int row = 0; row < 3; ++row )
    {
        for ( int col = 0; col < 3; ++col )
        {
            camera.rotation[ row * 3 + col ] = rotation( row, col );
        }
        camera.translation[ row ] = img_ptr->m_pose_c2w_t( row );
    }

    image.data = img_ptr->m_img.data;
    image.row_stride = static_cast< int >( img_ptr->m_img.step );
    image.channels = img_ptr->m_img.channels();

    if ( c_colorize_select_point( &camera, &image, 1, world_point, C_COLORIZE_SAMPLE_BILINEAR,
                                  C_COLORIZE_SELECT_NEAREST_DISTANCE, &c_result ) == 0 )
    {
        result.failure_reason = static_cast< ProjectionFailureReason >( c_result.failure_reason );
        return result;
    }

    result.success = true;
    result.failure_reason = ProjectionFailureReason::kNone;
    result.camera_index = c_result.camera_index;
    result.u = c_result.u;
    result.v = c_result.v;
    result.camera_distance = ( pt_w - img_ptr->m_pose_w2c_t ).norm();
    result.rgb = vec_3( c_result.bgr[ 0 ], c_result.bgr[ 1 ], c_result.bgr[ 2 ] );
    return result;
}

void record_colorize_comparison( const ColorizeExecutionResult &c_result, const ColorizeExecutionResult &legacy_result,
                                 ColorizeDebugStats *stats )
{
    if ( stats == nullptr )
    {
        return;
    }
    if ( c_result.success != legacy_result.success )
    {
        stats->legacy_projection_disagreement++;
        return;
    }
    if ( c_result.success == false )
    {
        return;
    }
    const double diff_b = std::abs( c_result.rgb( 0 ) - legacy_result.rgb( 0 ) );
    const double diff_g = std::abs( c_result.rgb( 1 ) - legacy_result.rgb( 1 ) );
    const double diff_r = std::abs( c_result.rgb( 2 ) - legacy_result.rgb( 2 ) );
    const double diff_mean = ( diff_b + diff_g + diff_r ) / 3.0;
    const double diff_max = std::max( diff_b, std::max( diff_g, diff_r ) );
    stats->legacy_color_diff_samples++;
    stats->legacy_color_diff_sum += diff_mean;
    stats->legacy_color_diff_max = std::max( stats->legacy_color_diff_max, diff_max );
}

ColorizeExecutionResult run_active_colorize( const std::shared_ptr< Image_frame > &img_ptr, const vec_3 &pt_w,
                                             ColorizeDebugStats *stats )
{
#if R3LIVE_USE_C_COLORIZE
    ColorizeExecutionResult active_result = run_c_colorize( img_ptr, pt_w );
    ColorizeExecutionResult legacy_result;
#if R3LIVE_VERIFY_C_COLORIZE
    legacy_result = run_legacy_colorize( img_ptr, pt_w );
    record_colorize_comparison( active_result, legacy_result, stats );
#endif
#else
    ColorizeExecutionResult active_result = run_legacy_colorize( img_ptr, pt_w );
#if R3LIVE_VERIFY_C_COLORIZE
    ColorizeExecutionResult c_result = run_c_colorize( img_ptr, pt_w );
    record_colorize_comparison( c_result, active_result, stats );
#endif
#endif

    if ( stats != nullptr )
    {
        stats->record_result( active_result );
    }
    return active_result;
}

long apply_color_to_point( const std::shared_ptr< RGB_pts > &rgb_pt, const ColorizeExecutionResult &result,
                           const double obs_time )
{
    if ( result.success == false )
    {
        return 0;
    }

    const long rgb_updated = rgb_pt->update_rgb( result.rgb, result.camera_distance,
                                                 vec_3( image_obs_cov, image_obs_cov, image_obs_cov ), obs_time );
    return rgb_updated;
}

void apply_gray_to_point( const std::shared_ptr< Image_frame > &img_ptr, const std::shared_ptr< RGB_pts > &rgb_pt,
                          const ColorizeExecutionResult &result )
{
    if ( result.success == false )
    {
        return;
    }

    vec_2  gama_bak = img_ptr->m_gama_para;
    double u = result.u;
    double v = result.v;
    img_ptr->m_gama_para = vec_2( 1.0, 0.0 );
    const double gray = img_ptr->get_grey_color( u, v, 0 );
    rgb_pt->update_gray( gray, result.camera_distance );
    img_ptr->m_gama_para = gama_bak;
}

void log_colorize_stats( const char *tag, const std::shared_ptr< Image_frame > &img_ptr, const ColorizeDebugStats &stats,
                         const long render_updates )
{
#if R3LIVE_VERIFY_C_COLORIZE
    std::ostringstream oss;
    oss << "[" << tag << "] frame=" << img_ptr->m_frame_idx << " proj_ok=" << stats.projection_success
        << " proj_fail=" << stats.projection_fail << " behind=" << stats.behind_camera << " oob=" << stats.out_of_bounds
        << " invalid=" << stats.invalid_input << " render_updates=" << render_updates << " per_camera=[";
    for ( size_t idx = 0; idx < stats.per_camera_hits.size(); ++idx )
    {
        if ( idx )
        {
            oss << ",";
        }
        oss << idx << ":" << stats.per_camera_hits[ idx ];
    }
    oss << "]";
    if ( stats.legacy_color_diff_samples > 0 )
    {
        oss << " legacy_diff_mean=" << ( stats.legacy_color_diff_sum / stats.legacy_color_diff_samples )
            << " legacy_diff_max=" << stats.legacy_color_diff_max;
    }
    oss << " legacy_proj_diff=" << stats.legacy_projection_disagreement;
    scope_color( ANSI_COLOR_CYAN_BOLD );
    cout << oss.str() << ANSI_COLOR_RESET << endl;
#else
    (void)tag;
    (void)img_ptr;
    (void)stats;
    (void)render_updates;
#endif
}
} // namespace

void RGB_pts::set_pos(const vec_3 &pos)
{
    m_pos[0] = pos(0);
    m_pos[1] = pos(1);
    m_pos[2] = pos(2);
}

vec_3 RGB_pts::get_pos()
{
    return vec_3(m_pos[0], m_pos[1], m_pos[2]);
}

mat_3_3 RGB_pts::get_rgb_cov()
{
    mat_3_3 cov_mat = mat_3_3::Zero();
    for (int i = 0; i < 3; i++)
    {
        cov_mat(i, i) = m_cov_rgb[i];
    }
    return cov_mat;
}

vec_3 RGB_pts::get_rgb()
{
    return vec_3(m_rgb[0], m_rgb[1], m_rgb[2]);
}

pcl::PointXYZI RGB_pts::get_pt()
{
    pcl::PointXYZI pt;
    pt.x = m_pos[0];
    pt.y = m_pos[1];
    pt.z = m_pos[2];
    return pt;
}

void RGB_pts::update_gray(const double gray, const double obs_dis)
{
    if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
    {
        return;
    }
    m_gray = (m_gray * m_N_gray + gray) / (m_N_gray + 1);
    if (m_obs_dis == 0 || (obs_dis < m_obs_dis))
    {
        m_obs_dis = obs_dis;
        // m_gray = gray;
    }
    m_N_gray++;
    // TODO: cov update
};

const double image_obs_cov = 15;
const double process_noise_sigma = 0.1;

int RGB_pts::update_rgb(const vec_3 &rgb, const double obs_dis, const vec_3 obs_sigma, const double obs_time)
{
    if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
    {
        return 0;
    }

    if( m_N_rgb == 0)
    {
        // For first time of observation.
        m_last_obs_time = obs_time;
        m_obs_dis = obs_dis;
        for (int i = 0; i < 3; i++)
        {
            m_rgb[i] = rgb[i];
            m_cov_rgb[i] = obs_sigma(i) ;
        }
        m_N_rgb = 1;
        return 0;
    }
    // State estimation for robotics, section 2.2.6, page 37-38
    for(int i = 0 ; i < 3; i++)
    {
        m_cov_rgb[i] = (m_cov_rgb[i] + process_noise_sigma * (obs_time - m_last_obs_time)); // Add process noise
        double old_sigma = m_cov_rgb[i];
        m_cov_rgb[i] = sqrt( 1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)) );
        m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * ( m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i) );
    }

    if (obs_dis < m_obs_dis)
    {
        m_obs_dis = obs_dis;
    }
    m_last_obs_time = obs_time;
    m_N_rgb++;
    return 1;
}

void Global_map::clear()
{
    m_rgb_pts_vec.clear();
}

void Global_map::set_minmum_dis(double minimum_dis)
{
    m_hashmap_3d_pts.clear();
    m_minimum_pts_size = minimum_dis;
}

Global_map::Global_map( int if_start_service )
{
    m_mutex_pts_vec = std::make_shared< std::mutex >();
    m_mutex_img_pose_for_projection = std::make_shared< std::mutex >();
    m_mutex_recent_added_list = std::make_shared< std::mutex >();
    m_mutex_rgb_pts_in_recent_hitted_boxes = std::make_shared< std::mutex >();
    m_mutex_m_box_recent_hitted = std::make_shared< std::mutex >();
    m_mutex_pts_last_visited = std::make_shared< std::mutex >();
    // Allocate memory for pointclouds
    if ( Common_tools::get_total_phy_RAM_size_in_GB() < 12 )
    {
        scope_color( ANSI_COLOR_RED_BOLD );
        std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << "I have detected your physical memory smaller than 12GB (currently: " << Common_tools::get_total_phy_RAM_size_in_GB()
             << "GB). I recommend you to add more physical memory for improving the overall performance of R3LIVE." << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
        m_rgb_pts_vec.reserve( 1e8 );
    }
    else
    {
        m_rgb_pts_vec.reserve( 1e9 );
    }
    // m_rgb_pts_in_recent_visited_voxels.reserve( 1e6 );
    if ( if_start_service )
    {
        m_thread_service = std::make_shared< std::thread >( &Global_map::service_refresh_pts_for_projection, this );
    }
}
Global_map::~Global_map(){};

void Global_map::service_refresh_pts_for_projection()
{
    eigen_q last_pose_q = eigen_q::Identity();
    Common_tools::Timer                timer;
    std::shared_ptr< Image_frame > img_for_projection = std::make_shared< Image_frame >();
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        m_mutex_img_pose_for_projection->lock();
         
        *img_for_projection = m_img_for_projection;
        m_mutex_img_pose_for_projection->unlock();
        if (img_for_projection->m_img_cols == 0 || img_for_projection->m_img_rows == 0)
        {
            continue;
        }

        if (img_for_projection->m_frame_idx == m_updated_frame_index)
        {
            continue;
        }
        timer.tic(" ");
        std::shared_ptr<std::vector<std::shared_ptr<RGB_pts>>> pts_rgb_vec_for_projection = std::make_shared<std::vector<std::shared_ptr<RGB_pts>>>();
        if (m_if_get_all_pts_in_boxes_using_mp)
        {
            std::vector<std::shared_ptr<RGB_pts>>  pts_in_recent_hitted_boxes;
            pts_in_recent_hitted_boxes.reserve(1e6);
            std::unordered_set< std::shared_ptr< RGB_Voxel> > boxes_recent_hitted;
            m_mutex_m_box_recent_hitted->lock();
            boxes_recent_hitted = m_voxels_recent_visited;
            m_mutex_m_box_recent_hitted->unlock();

            // get_all_pts_in_boxes(boxes_recent_hitted, pts_in_recent_hitted_boxes);
            m_mutex_rgb_pts_in_recent_hitted_boxes->lock();
            // m_rgb_pts_in_recent_visited_voxels = pts_in_recent_hitted_boxes;
            m_mutex_rgb_pts_in_recent_hitted_boxes->unlock();
        }
        selection_points_for_projection(img_for_projection, pts_rgb_vec_for_projection.get(), nullptr, 10.0, 1);
        m_mutex_pts_vec->lock();
        m_pts_rgb_vec_for_projection = pts_rgb_vec_for_projection;
        m_updated_frame_index = img_for_projection->m_frame_idx;
        // cout << ANSI_COLOR_MAGENTA_BOLD << "Refresh pts_for_projection size = " << m_pts_rgb_vec_for_projection->size()
        //      << " | " << m_rgb_pts_vec.size()
        //      << ", cost time = " << timer.toc() << ANSI_COLOR_RESET << endl;
        m_mutex_pts_vec->unlock();
        last_pose_q = img_for_projection->m_pose_w2c_q;
    }
}

void Global_map::render_points_for_projection(std::shared_ptr<Image_frame> &img_ptr)
{
    m_mutex_pts_vec->lock();
    if (m_pts_rgb_vec_for_projection != nullptr)
    {
        render_pts_in_voxels(img_ptr, *m_pts_rgb_vec_for_projection);
        // render_pts_in_voxels(img_ptr, m_rgb_pts_vec);
    }
    m_last_updated_frame_idx = img_ptr->m_frame_idx;
    m_mutex_pts_vec->unlock();
}

void Global_map::update_pose_for_projection(std::shared_ptr<Image_frame> &img, double fov_margin)
{
    m_mutex_img_pose_for_projection->lock();
    m_img_for_projection.set_intrinsic(img->m_cam_K);
    m_img_for_projection.m_img_cols = img->m_img_cols;
    m_img_for_projection.m_img_rows = img->m_img_rows;
    m_img_for_projection.m_fov_margin = fov_margin;
    m_img_for_projection.m_frame_idx = img->m_frame_idx;
    m_img_for_projection.m_pose_w2c_q = img->m_pose_w2c_q;
    m_img_for_projection.m_pose_w2c_t = img->m_pose_w2c_t;
    m_img_for_projection.m_img_gray = img->m_img_gray; // clone?
    m_img_for_projection.m_img = img->m_img;           // clone?
    m_img_for_projection.refresh_pose_for_projection();
    m_mutex_img_pose_for_projection->unlock();
}

bool Global_map::is_busy()
{
    return m_in_appending_pts;
}

template int Global_map::append_points_to_global_map<pcl::PointXYZI>(pcl::PointCloud<pcl::PointXYZI> &pc_in, double  added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step);
template int Global_map::append_points_to_global_map<pcl::PointXYZRGB>(pcl::PointCloud<pcl::PointXYZRGB> &pc_in, double  added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step);

template <typename T>
int Global_map::append_points_to_global_map(pcl::PointCloud<T> &pc_in, double  added_time,  std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step)
{
    m_in_appending_pts = 1;
    Common_tools::Timer tim;
    tim.tic();
    int acc = 0;
    int rej = 0;
    if (pts_added_vec != nullptr)
    {
        pts_added_vec->clear();
    }
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited;
    if (m_recent_visited_voxel_activated_time == 0)
    {
        voxels_recent_visited.clear();
    }
    else
    {
        m_mutex_m_box_recent_hitted->lock();
        voxels_recent_visited = m_voxels_recent_visited;
        m_mutex_m_box_recent_hitted->unlock();
        for( Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end();  )
        {
            if ( added_time - ( *it )->m_last_visited_time > m_recent_visited_voxel_activated_time )
            {
                it = voxels_recent_visited.erase( it );
                continue;
            }
            it++;
        }
        cout << "Restored voxel number = " << voxels_recent_visited.size() << endl;
    }
    int number_of_voxels_before_add = voxels_recent_visited.size();
    int pt_size = pc_in.points.size();
    // step = 4;
    for (int pt_idx = 0; pt_idx < pt_size; pt_idx += step)
    {
        int add = 1;
        int grid_x = std::round(pc_in.points[pt_idx].x / m_minimum_pts_size);
        int grid_y = std::round(pc_in.points[pt_idx].y / m_minimum_pts_size);
        int grid_z = std::round(pc_in.points[pt_idx].z / m_minimum_pts_size);
        int box_x =  std::round(pc_in.points[pt_idx].x / m_voxel_resolution);
        int box_y =  std::round(pc_in.points[pt_idx].y / m_voxel_resolution);
        int box_z =  std::round(pc_in.points[pt_idx].z / m_voxel_resolution);
        if (m_hashmap_3d_pts.if_exist(grid_x, grid_y, grid_z))
        {
            add = 0;
            if (pts_added_vec != nullptr)
            {
                pts_added_vec->push_back(m_hashmap_3d_pts.m_map_3d_hash_map[grid_x][grid_y][grid_z]);
            }
        }
        RGB_voxel_ptr box_ptr;
        if(!m_hashmap_voxels.if_exist(box_x, box_y, box_z))
        {
            std::shared_ptr<RGB_Voxel> box_rgb = std::make_shared<RGB_Voxel>();
            m_hashmap_voxels.insert( box_x, box_y, box_z, box_rgb );
            box_ptr = box_rgb;
        }
        else
        {
            box_ptr = m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z];
        }
        voxels_recent_visited.insert( box_ptr );
        box_ptr->m_last_visited_time = added_time;
        if (add == 0)
        {
            rej++;
            continue;
        }
        acc++;
        std::shared_ptr<RGB_pts> pt_rgb = std::make_shared<RGB_pts>();
        pt_rgb->set_pos(vec_3(pc_in.points[pt_idx].x, pc_in.points[pt_idx].y, pc_in.points[pt_idx].z));
        pt_rgb->m_pt_index = m_rgb_pts_vec.size();
        m_rgb_pts_vec.push_back(pt_rgb);
        m_hashmap_3d_pts.insert(grid_x, grid_y, grid_z, pt_rgb);
        box_ptr->add_pt(pt_rgb);
        if (pts_added_vec != nullptr)
        {
            pts_added_vec->push_back(pt_rgb);
        }
    }
    m_in_appending_pts = 0;
    m_mutex_m_box_recent_hitted->lock();
    m_voxels_recent_visited = voxels_recent_visited ;
    m_mutex_m_box_recent_hitted->unlock();
    return (m_voxels_recent_visited.size() -  number_of_voxels_before_add);
}


void Global_map::render_pts_in_voxels(std::shared_ptr<Image_frame> &img_ptr, std::vector<std::shared_ptr<RGB_pts>> &pts_for_render, double obs_time)
{
    Common_tools::Timer tim;
    tim.tic();
    int pt_size = pts_for_render.size();
    long render_updates = 0;
    ColorizeDebugStats stats( 1 );
    m_last_updated_frame_idx = img_ptr->m_frame_idx;
    for (int i = 0; i < pt_size; i++)
    {
        vec_3 pt_w = pts_for_render[i]->get_pos();
        ColorizeExecutionResult colorized = run_active_colorize( img_ptr, pt_w, &stats );
        if ( colorized.success == false )
        {
            continue;
        }
        apply_gray_to_point( img_ptr, pts_for_render[ i ], colorized );
        render_updates += apply_color_to_point( pts_for_render[ i ], colorized, obs_time );
    }
    log_colorize_stats( "render_single", img_ptr, stats, render_updates );
}

Common_tools::Cost_time_logger cost_time_logger_render("/home/ziv/temp/render_thr.log");

static inline ThreadRenderReport thread_render_pts_in_voxel(const int & pt_start, const int & pt_end, const std::shared_ptr<Image_frame> & img_ptr,
                                                            const std::vector<RGB_voxel_ptr> * voxels_for_render, const double obs_time)
{
    ThreadRenderReport report( 1 );
    Common_tools::Timer tim;
    tim.tic();
    for (int voxel_idx = pt_start; voxel_idx < pt_end; voxel_idx++)
    {
        RGB_voxel_ptr voxel_ptr = (*voxels_for_render)[ voxel_idx ];
        for ( int pt_idx = 0; pt_idx < voxel_ptr->m_pts_in_grid.size(); pt_idx++ )
        {
            vec_3 pt_w = voxel_ptr->m_pts_in_grid[ pt_idx ]->get_pos();
            ColorizeExecutionResult colorized = run_active_colorize( img_ptr, pt_w, &report.stats );
            if ( colorized.success == false )
            {
                continue;
            }
            report.render_updates += apply_color_to_point( voxel_ptr->m_pts_in_grid[ pt_idx ], colorized, obs_time );
        }
    }
    report.cost_time = tim.toc() * 100;
    return report;
}

std::vector<RGB_voxel_ptr>  g_voxel_for_render;
void render_pts_in_voxels_mp(std::shared_ptr<Image_frame> &img_ptr, std::unordered_set<RGB_voxel_ptr> * _voxels_for_render,  const double & obs_time)
{
    Common_tools::Timer tim;
    g_voxel_for_render.clear();
    for(Voxel_set_iterator it = (*_voxels_for_render).begin(); it != (*_voxels_for_render).end(); it++)
    {
        g_voxel_for_render.push_back(*it);
    }
    tim.tic("Render_mp");
    int numbers_of_voxels = g_voxel_for_render.size();
    g_cost_time_logger.record("Pts_num", numbers_of_voxels);
    long render_updates = 0;
    ColorizeDebugStats total_stats( 1 );
    if(USING_OPENCV_TBB)
    {
        std::mutex stats_mutex;
        cv::parallel_for_(cv::Range(0, numbers_of_voxels), [&](const cv::Range &r)
                          {
                              ThreadRenderReport report = thread_render_pts_in_voxel(r.start, r.end, img_ptr, &g_voxel_for_render, obs_time);
                              std::lock_guard<std::mutex> lock(stats_mutex);
                              render_updates += report.render_updates;
                              total_stats.merge(report.stats);
                          });
    }
    else
    {
        int num_of_threads = std::min(8*2, (int)numbers_of_voxels);
        // results.clear();
        std::vector<std::future<ThreadRenderReport>> results;
        results.resize(num_of_threads);
        tim.tic("Com");
        for (int thr = 0; thr < num_of_threads; thr++)
        {
            // cv::Range range(thr * pt_size / num_of_threads, (thr + 1) * pt_size / num_of_threads);
            int start = thr * numbers_of_voxels / num_of_threads;
            int end = (thr + 1) * numbers_of_voxels / num_of_threads;
            results[thr] = m_thread_pool_ptr->commit_task(thread_render_pts_in_voxel, start, end,  img_ptr, &g_voxel_for_render, obs_time);
        }
        g_cost_time_logger.record(tim, "Com");
        tim.tic("wait_Opm");
        for (int thr = 0; thr < num_of_threads; thr++)
        {
            ThreadRenderReport report = results[thr].get();
            render_updates += report.render_updates;
            total_stats.merge( report.stats );
            cost_time_logger_render.record(std::string("T_").append(std::to_string(thr)), report.cost_time );
        }
        g_cost_time_logger.record(tim, "wait_Opm");
        cost_time_logger_render.record(tim, "wait_Opm");
    }
    // img_ptr->release_image();
    cost_time_logger_render.flush_d();
    g_cost_time_logger.record(tim, "Render_mp");
    g_cost_time_logger.record("Pts_num_r", render_updates);
    log_colorize_stats( "render_mp", img_ptr, total_stats, render_updates );
    
}

void Global_map::render_with_a_image(std::shared_ptr<Image_frame> &img_ptr, int if_select)
{

    std::vector<std::shared_ptr<RGB_pts>> pts_for_render;
    // pts_for_render = m_rgb_pts_vec;
    if (if_select)
    {
        selection_points_for_projection(img_ptr, &pts_for_render, nullptr, 1.0);
    }
    else
    {
        pts_for_render = m_rgb_pts_vec;
    }
    render_pts_in_voxels(img_ptr, pts_for_render);
}

void Global_map::selection_points_for_projection(std::shared_ptr<Image_frame> &image_pose, std::vector<std::shared_ptr<RGB_pts>> *pc_out_vec,
                                                            std::vector<cv::Point2f> *pc_2d_out_vec, double minimum_dis,
                                                            int skip_step,
                                                            int use_all_pts)
{
    Common_tools::Timer tim;
    tim.tic();
    if (pc_out_vec != nullptr)
    {
        pc_out_vec->clear();
    }
    if (pc_2d_out_vec != nullptr)
    {
        pc_2d_out_vec->clear();
    }
    Hash_map_2d<int, int> mask_index;
    Hash_map_2d<int, float> mask_depth;

    std::map<int, cv::Point2f> map_idx_draw_center;
    std::map<int, cv::Point2f> map_idx_draw_center_raw_pose;

    int u, v;
    double u_f, v_f;
    // cv::namedWindow("Mask", cv::WINDOW_FREERATIO);
    int acc = 0;
    int blk_rej = 0;
    // int pts_size = m_rgb_pts_vec.size();
    std::vector<std::shared_ptr<RGB_pts>> pts_for_projection;
    m_mutex_m_box_recent_hitted->lock();
    std::unordered_set< std::shared_ptr< RGB_Voxel > > boxes_recent_hitted = m_voxels_recent_visited;
    m_mutex_m_box_recent_hitted->unlock();
    if ( (!use_all_pts) && boxes_recent_hitted.size())
    {
        m_mutex_rgb_pts_in_recent_hitted_boxes->lock();
        
        for(Voxel_set_iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++)
        {
            // pts_for_projection.push_back( (*it)->m_pts_in_grid.back() );
            if ( ( *it )->m_pts_in_grid.size() )
            {
                 pts_for_projection.push_back( (*it)->m_pts_in_grid.back() );
                // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ 0 ] );
                // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ ( *it )->m_pts_in_grid.size()-1 ] );
            }
        }

        m_mutex_rgb_pts_in_recent_hitted_boxes->unlock();
    }
    else
    {
        pts_for_projection = m_rgb_pts_vec;
    }
    int pts_size = pts_for_projection.size();
    for (int pt_idx = 0; pt_idx < pts_size; pt_idx += skip_step)
    {
        vec_3 pt = pts_for_projection[pt_idx]->get_pos();
        double depth = (pt - image_pose->m_pose_w2c_t).norm();
        if (depth > m_maximum_depth_for_projection)
        {
            continue;
        }
        if (depth < m_minimum_depth_for_projection)
        {
            continue;
        }
        bool res = image_pose->project_3d_point_in_this_img(pt, u_f, v_f, nullptr, 1.0);
        if (res == false)
        {
            continue;
        }
        u = std::round(u_f / minimum_dis) * minimum_dis; // Why can not work
        v = std::round(v_f / minimum_dis) * minimum_dis;
        if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth)
        {
            acc++;
            if (mask_index.if_exist(u, v))
            {
                // erase old point
                int old_idx = mask_index.m_map_2d_hash_map[u][v];
                blk_rej++;
                map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
                map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
            }
            mask_index.m_map_2d_hash_map[u][v] = (int)pt_idx;
            mask_depth.m_map_2d_hash_map[u][v] = (float)depth;
            map_idx_draw_center[pt_idx] = cv::Point2f(v, u);
            map_idx_draw_center_raw_pose[pt_idx] = cv::Point2f(u_f, v_f);
        }
    }

    if (pc_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
        {
            // pc_out_vec->push_back(m_rgb_pts_vec[it->first]);
            pc_out_vec->push_back(pts_for_projection[it->first]);
        }
    }

    if (pc_2d_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
        {
            pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
        }
    }

}

void Global_map::save_to_pcd(std::string dir_name, std::string _file_name, int save_pts_with_views )
{
    Common_tools::Timer tim;
    Common_tools::create_dir(dir_name);
    std::string file_name = std::string(dir_name).append(_file_name);
    scope_color(ANSI_COLOR_BLUE_BOLD);
    cout << "Save Rgb points to " << file_name << endl;
    fflush(stdout);
    pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
    long pt_size = m_rgb_pts_vec.size();
    pc_rgb.resize(pt_size);
    long pt_count = 0;
    for (long i = pt_size - 1; i > 0; i--)
    //for (int i = 0; i  <  pt_size; i++)
    {
        if ( i % 1000 == 0)
        {
            cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map " << (int)( (pt_size- 1 -i ) * 100.0 / (pt_size-1) ) << " % ...";
            fflush(stdout);
        }

        if (m_rgb_pts_vec[i]->m_N_rgb < save_pts_with_views)
        {
            continue;
        }
        pcl::PointXYZRGB pt;
        pc_rgb.points[ pt_count ].x = m_rgb_pts_vec[ i ]->m_pos[ 0 ];
        pc_rgb.points[ pt_count ].y = m_rgb_pts_vec[ i ]->m_pos[ 1 ];
        pc_rgb.points[ pt_count ].z = m_rgb_pts_vec[ i ]->m_pos[ 2 ];
        pc_rgb.points[ pt_count ].r = m_rgb_pts_vec[ i ]->m_rgb[ 2 ];
        pc_rgb.points[ pt_count ].g = m_rgb_pts_vec[ i ]->m_rgb[ 1 ];
        pc_rgb.points[ pt_count ].b = m_rgb_pts_vec[ i ]->m_rgb[ 0 ];
        pt_count++;
    }
    cout << ANSI_DELETE_CURRENT_LINE  << "Saving offline map 100% ..." << endl;
    pc_rgb.resize(pt_count);
    cout << "Total have " << pt_count << " points." << endl;
    tim.tic();
    cout << "Now write to: " << file_name << endl; 
    pcl::io::savePCDFileBinary(std::string(file_name).append(".pcd"), pc_rgb);
    cout << "Save PCD cost time = " << tim.toc() << endl;
}

void Global_map::save_and_display_pointcloud(std::string dir_name, std::string file_name, int save_pts_with_views)
{
    save_to_pcd(dir_name, file_name, save_pts_with_views);
    scope_color(ANSI_COLOR_WHITE_BOLD);
    cout << "========================================================" << endl;
    cout << "Open pcl_viewer to display point cloud, close the viewer's window to continue mapping process ^_^" << endl;
    cout << "========================================================" << endl;
    system(std::string("pcl_viewer ").append(dir_name).append("/rgb_pt.pcd").c_str());
}
