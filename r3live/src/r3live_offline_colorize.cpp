#include <algorithm>
#include <cstdint>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/io/pcd_io.h>

#include "tools_color_printf.hpp"
#include "tools_data_io.hpp"
#include "rgb_map/image_frame.hpp"
#include "rgb_map/pointcloud_rgbd.hpp"
#include "rgb_map/offline_map_recorder.hpp"
#include "c_colorize/colorize.h"

namespace
{
const double kOfflineImageObsCov = 15.0;

struct OfflineCameraConfig
{
    std::string                                        name;
    Eigen::Matrix3d                                    intrinsic = Eigen::Matrix3d::Identity();
    Eigen::Matrix< double, 5, 1 >                      dist_coeffs = Eigen::Matrix< double, 5, 1 >::Zero();
    Eigen::Matrix3d                                    lidar_ext_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d                                    lidar_ext_t = Eigen::Vector3d::Zero();
    int                                                image_width = 0;
    int                                                image_height = 0;
    cv::Mat                                            undistort_map1;
    cv::Mat                                            undistort_map2;
    cv::Size                                           undistort_size;
};

struct OfflineFrameEntry
{
    double                     timestamp = 0.0;
    std::string                frame_id;
    std::string                pcd_path;
    std::string                pose_path;
    Eigen::Vector3d            lidar_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond         lidar_q = Eigen::Quaterniond::Identity();
    std::vector< std::string > image_paths;
};

struct OfflineStats
{
    long              projection_success = 0;
    long              projection_fail = 0;
    long              out_of_bounds = 0;
    long              behind_camera = 0;
    long              invalid_input = 0;
    long              rgb_updates = 0;
    std::vector<long> per_camera_hits;

    explicit OfflineStats( size_t camera_count = 0 ) : per_camera_hits( camera_count, 0 ) {}

    void merge( const OfflineStats &other )
    {
        projection_success += other.projection_success;
        projection_fail += other.projection_fail;
        out_of_bounds += other.out_of_bounds;
        behind_camera += other.behind_camera;
        invalid_input += other.invalid_input;
        rgb_updates += other.rgb_updates;
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

struct OfflineAppConfig
{
    std::string               config_path;
    std::string               dataset_root;
    std::string               input_mode = "directory_layout";
    std::string               frame_list = "frames.txt";
    std::string               pose_dir = "pos";
    std::string               pcd_dir = "pcd";
    std::string               image_extension = ".jpg";
    std::string               output_dir;
    std::string               camera_selection_mode = "nearest_distance";
    std::string               sample_mode = "bilinear";
    std::vector< std::string > camera_names;
    int                       append_global_map_point_step = 1;
    int                       save_frame_colored_pcd = 1;
    int                       save_offline_map = 0;
    int                       use_frame_pose = 1;
    int                       frame_point_in_world = 0;
    double                    minimum_pts_size = 0.05;
};

std::string trim_copy( const std::string &input )
{
    const std::string whitespaces = " \t\r\n";
    const size_t      begin = input.find_first_not_of( whitespaces );
    if ( begin == std::string::npos )
    {
        return std::string();
    }
    const size_t end = input.find_last_not_of( whitespaces );
    return input.substr( begin, end - begin + 1 );
}

std::string resolve_path( const std::string &dataset_root, const std::string &path )
{
    if ( path.empty() )
    {
        return path;
    }
    if ( path[ 0 ] == '/' )
    {
        return path;
    }
    if ( dataset_root.empty() )
    {
        return path;
    }
    if ( dataset_root.back() == '/' )
    {
        return dataset_root + path;
    }
    return dataset_root + "/" + path;
}

std::string get_file_stem( const std::string &path )
{
    const size_t slash_pos = path.find_last_of( "/\\" );
    const size_t begin = ( slash_pos == std::string::npos ) ? 0 : slash_pos + 1;
    const size_t dot_pos = path.find_last_of( '.' );
    if ( dot_pos == std::string::npos || dot_pos < begin )
    {
        return path.substr( begin );
    }
    return path.substr( begin, dot_pos - begin );
}

bool is_regular_file( const std::string &path )
{
    struct stat st;
    if ( stat( path.c_str(), &st ) != 0 )
    {
        return false;
    }
    return S_ISREG( st.st_mode );
}

bool has_suffix( const std::string &value, const std::string &suffix )
{
    if ( value.size() < suffix.size() )
    {
        return false;
    }
    return value.compare( value.size() - suffix.size(), suffix.size(), suffix ) == 0;
}

std::vector< std::string > list_files_with_suffix( const std::string &dir_path, const std::string &suffix )
{
    std::vector< std::string > file_paths;
    DIR *                      dir = opendir( dir_path.c_str() );
    if ( dir == nullptr )
    {
        return file_paths;
    }

    while ( true )
    {
        dirent *entry = readdir( dir );
        if ( entry == nullptr )
        {
            break;
        }

        const std::string file_name = entry->d_name;
        if ( file_name == "." || file_name == ".." )
        {
            continue;
        }
        if ( has_suffix( file_name, suffix ) == false )
        {
            continue;
        }
        const std::string full_path = resolve_path( dir_path, file_name );
        if ( is_regular_file( full_path ) )
        {
            file_paths.push_back( full_path );
        }
    }

    closedir( dir );
    std::sort( file_paths.begin(), file_paths.end() );
    return file_paths;
}

bool load_pose_file( const std::string &pose_path, OfflineFrameEntry &frame )
{
    Eigen::MatrixXd pose_data = Common_tools::load_mat_from_txt< double >( pose_path );
    std::vector< double > values;
    for ( int row = 0; row < pose_data.rows(); ++row )
    {
        for ( int col = 0; col < pose_data.cols(); ++col )
        {
            values.push_back( pose_data( row, col ) );
        }
    }

    if ( values.size() == 7 )
    {
        frame.lidar_q = Eigen::Quaterniond( values[ 0 ], values[ 1 ], values[ 2 ], values[ 3 ] );
        frame.lidar_t << values[ 4 ], values[ 5 ], values[ 6 ];
    }
    else if ( values.size() == 8 )
    {
        frame.timestamp = values[ 0 ];
        frame.lidar_q = Eigen::Quaterniond( values[ 1 ], values[ 2 ], values[ 3 ], values[ 4 ] );
        frame.lidar_t << values[ 5 ], values[ 6 ], values[ 7 ];
    }
    else
    {
        cout << ANSI_COLOR_RED_BOLD << "Unsupported pose format in " << pose_path
             << ", expected 7 values (qw qx qy qz tx ty tz) or 8 values (timestamp qw qx qy qz tx ty tz)." << ANSI_COLOR_RESET
             << endl;
        return false;
    }

    frame.lidar_q.normalize();
    return true;
}

bool read_string_array( const cv::FileNode &node, std::vector< std::string > &values )
{
    if ( node.empty() || node.isSeq() == false )
    {
        return false;
    }

    values.clear();
    for ( cv::FileNodeIterator it = node.begin(); it != node.end(); ++it )
    {
        values.push_back( ( std::string )*it );
    }
    return true;
}

bool read_double_array( const cv::FileNode &node, std::vector< double > &values )
{
    if ( node.empty() || node.isSeq() == false )
    {
        return false;
    }

    values.clear();
    for ( cv::FileNodeIterator it = node.begin(); it != node.end(); ++it )
    {
        values.push_back( ( double )*it );
    }
    return true;
}

template < typename T >
void read_scalar_or_default( const cv::FileNode &parent, const std::string &key, T &value )
{
    const cv::FileNode node = parent[ key ];
    if ( node.empty() == false )
    {
        node >> value;
    }
}

bool load_camera_config( const cv::FileNode &offline_node, const std::string &camera_name, OfflineCameraConfig &camera )
{
    std::vector< double > intrinsic_data, dist_coeffs_data, ext_R_data, ext_t_data;
    const cv::FileNode    camera_node = offline_node[ "cameras" ][ camera_name ];
    camera.name = camera_name;
    if ( camera_node.empty() )
    {
        cout << ANSI_COLOR_RED_BOLD << "Missing camera config for " << camera_name << ANSI_COLOR_RESET << endl;
        return false;
    }

    read_double_array( camera_node[ "camera_intrinsic" ], intrinsic_data );
    read_double_array( camera_node[ "camera_dist_coeffs" ], dist_coeffs_data );
    read_double_array( camera_node[ "lidar_ext_R" ], ext_R_data );
    read_double_array( camera_node[ "lidar_ext_t" ], ext_t_data );
    read_scalar_or_default( camera_node, "image_width", camera.image_width );
    read_scalar_or_default( camera_node, "image_height", camera.image_height );

    if ( intrinsic_data.size() != 9 || dist_coeffs_data.size() != 5 || ext_R_data.size() != 9 || ext_t_data.size() != 3 )
    {
        cout << ANSI_COLOR_RED_BOLD << "Invalid camera config for " << camera_name << ANSI_COLOR_RESET << endl;
        return false;
    }

    camera.intrinsic = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( intrinsic_data.data() );
    camera.dist_coeffs = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( dist_coeffs_data.data() );
    camera.lidar_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( ext_R_data.data() );
    camera.lidar_ext_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( ext_t_data.data() );
    return true;
}

bool load_app_config( const std::string &config_path, OfflineAppConfig &config )
{
    cv::FileStorage fs( config_path, cv::FileStorage::READ );
    if ( fs.isOpened() == false )
    {
        cout << ANSI_COLOR_RED_BOLD << "Failed to open config file: " << config_path << ANSI_COLOR_RESET << endl;
        return false;
    }

    const cv::FileNode offline_node = fs[ "offline_colorize" ];
    if ( offline_node.empty() )
    {
        cout << ANSI_COLOR_RED_BOLD << "Missing 'offline_colorize' root node in " << config_path << ANSI_COLOR_RESET << endl;
        return false;
    }

    config.config_path = config_path;
    read_scalar_or_default( offline_node, "dataset_root", config.dataset_root );
    read_scalar_or_default( offline_node, "input_mode", config.input_mode );
    read_scalar_or_default( offline_node, "frame_list", config.frame_list );
    read_scalar_or_default( offline_node, "pose_dir", config.pose_dir );
    read_scalar_or_default( offline_node, "pcd_dir", config.pcd_dir );
    read_scalar_or_default( offline_node, "image_extension", config.image_extension );
    read_scalar_or_default( offline_node, "output_dir", config.output_dir );
    read_scalar_or_default( offline_node, "camera_selection_mode", config.camera_selection_mode );
    read_scalar_or_default( offline_node, "sample_mode", config.sample_mode );
    read_scalar_or_default( offline_node, "append_global_map_point_step", config.append_global_map_point_step );
    read_scalar_or_default( offline_node, "save_frame_colored_pcd", config.save_frame_colored_pcd );
    read_scalar_or_default( offline_node, "save_offline_map", config.save_offline_map );
    read_scalar_or_default( offline_node, "use_frame_pose", config.use_frame_pose );
    read_scalar_or_default( offline_node, "frame_point_in_world", config.frame_point_in_world );
    read_scalar_or_default( offline_node, "minimum_pts_size", config.minimum_pts_size );
    read_string_array( offline_node[ "camera_names" ], config.camera_names );

    if ( config.output_dir.empty() )
    {
        config.output_dir = std::string( Common_tools::get_home_folder() ).append( "/r3live_offline_output" );
    }
    return true;
}

void print_usage( const char *argv0 )
{
    cout << "Usage: " << argv0 << " [config.yaml] [dataset_root_override]" << endl;
}

bool load_point_cloud_xyzi( const std::string &pcd_path, pcl::PointCloud< pcl::PointXYZI > &cloud )
{
    pcl::PointCloud< pcl::PointXYZ >    cloud_xyz;
    pcl::PointCloud< pcl::PointXYZRGB > cloud_xyzrgb;
    if ( pcl::io::loadPCDFile( pcd_path, cloud ) == 0 )
    {
        return true;
    }
    if ( pcl::io::loadPCDFile( pcd_path, cloud_xyz ) == 0 )
    {
        cloud.clear();
        cloud.reserve( cloud_xyz.size() );
        for ( size_t idx = 0; idx < cloud_xyz.size(); ++idx )
        {
            pcl::PointXYZI pt;
            pt.x = cloud_xyz.points[ idx ].x;
            pt.y = cloud_xyz.points[ idx ].y;
            pt.z = cloud_xyz.points[ idx ].z;
            pt.intensity = 0;
            cloud.push_back( pt );
        }
        return true;
    }
    if ( pcl::io::loadPCDFile( pcd_path, cloud_xyzrgb ) == 0 )
    {
        cloud.clear();
        cloud.reserve( cloud_xyzrgb.size() );
        for ( size_t idx = 0; idx < cloud_xyzrgb.size(); ++idx )
        {
            pcl::PointXYZI pt;
            pt.x = cloud_xyzrgb.points[ idx ].x;
            pt.y = cloud_xyzrgb.points[ idx ].y;
            pt.z = cloud_xyzrgb.points[ idx ].z;
            pt.intensity = 0;
            cloud.push_back( pt );
        }
        return true;
    }
    return false;
}

void transform_cloud_to_world( const pcl::PointCloud< pcl::PointXYZI > &cloud_in, const Eigen::Quaterniond &lidar_q,
                               const Eigen::Vector3d &lidar_t, pcl::PointCloud< pcl::PointXYZI > &cloud_out )
{
    const Eigen::Matrix3d lidar_R = lidar_q.toRotationMatrix();
    cloud_out.clear();
    cloud_out.reserve( cloud_in.size() );
    for ( size_t idx = 0; idx < cloud_in.size(); ++idx )
    {
        Eigen::Vector3d       pt_lidar( cloud_in.points[ idx ].x, cloud_in.points[ idx ].y, cloud_in.points[ idx ].z );
        const Eigen::Vector3d pt_world = lidar_R * pt_lidar + lidar_t;
        pcl::PointXYZI        pt = cloud_in.points[ idx ];
        pt.x = pt_world( 0 );
        pt.y = pt_world( 1 );
        pt.z = pt_world( 2 );
        cloud_out.push_back( pt );
    }
}

void prepare_undistort_maps( OfflineCameraConfig &camera, const cv::Size &image_size )
{
    if ( camera.undistort_size == image_size && !camera.undistort_map1.empty() && !camera.undistort_map2.empty() )
    {
        return;
    }
    cv::Mat intrinsic_cv, dist_coeffs_cv;
    cv::eigen2cv( camera.intrinsic, intrinsic_cv );
    cv::eigen2cv( camera.dist_coeffs, dist_coeffs_cv );
    cv::initUndistortRectifyMap( intrinsic_cv, dist_coeffs_cv, cv::Mat(), intrinsic_cv, image_size, CV_16SC2,
                                 camera.undistort_map1, camera.undistort_map2 );
    camera.undistort_size = image_size;
}

bool load_frame_image( OfflineCameraConfig &camera, const OfflineFrameEntry &frame, const std::string &image_path,
                       std::shared_ptr< Image_frame > &image_frame )
{
    cv::Mat          raw = cv::imread( image_path, cv::IMREAD_COLOR );
    Eigen::Matrix3d  intrinsic = camera.intrinsic;
    cv::Size         image_size;
    std::shared_ptr< Image_frame > img_ptr = std::make_shared< Image_frame >( intrinsic );
    if ( raw.empty() )
    {
        return false;
    }
    image_size = cv::Size( raw.cols, raw.rows );
    if ( camera.image_width > 0 && camera.image_height > 0 )
    {
        image_size = cv::Size( camera.image_width, camera.image_height );
    }
    prepare_undistort_maps( camera, image_size );

    const Eigen::Matrix3d lidar_R = frame.lidar_q.toRotationMatrix();
    const Eigen::Matrix3d camera_R = lidar_R * camera.lidar_ext_R;
    const Eigen::Vector3d camera_t = lidar_R * camera.lidar_ext_t + frame.lidar_t;

    img_ptr->set_pose( eigen_q( camera_R ), camera_t );
    img_ptr->m_timestamp = frame.timestamp;
    img_ptr->m_raw_img = raw.clone();
    if ( raw.size() != image_size )
    {
        cv::resize( raw, raw, image_size );
    }
    cv::remap( raw, img_ptr->m_img, camera.undistort_map1, camera.undistort_map2, cv::INTER_LINEAR );
    img_ptr->m_img_rows = img_ptr->m_img.rows;
    img_ptr->m_img_cols = img_ptr->m_img.cols;
    img_ptr->init_cubic_interpolation();
    img_ptr->image_equalize();
    image_frame = img_ptr;
    return true;
}

void fill_colorize_inputs( const std::vector< std::shared_ptr< Image_frame > > &images, std::vector< c_colorize_camera_t > &cameras,
                           std::vector< c_colorize_image_t > &image_views )
{
    cameras.resize( images.size() );
    image_views.resize( images.size() );
    for ( size_t idx = 0; idx < images.size(); ++idx )
    {
        c_colorize_camera_t &camera = cameras[ idx ];
        c_colorize_image_t & image = image_views[ idx ];
        const Eigen::Matrix3d rotation = images[ idx ]->m_pose_c2w_q.toRotationMatrix();

        memset( &camera, 0, sizeof( camera ) );
        memset( &image, 0, sizeof( image ) );
        camera.fx = images[ idx ]->fx;
        camera.fy = images[ idx ]->fy;
        camera.cx = images[ idx ]->cx;
        camera.cy = images[ idx ]->cy;
        camera.image_rows = images[ idx ]->m_img.rows;
        camera.image_cols = images[ idx ]->m_img.cols;
        camera.fov_margin = images[ idx ]->m_fov_margin;
        for ( int row = 0; row < 3; ++row )
        {
            for ( int col = 0; col < 3; ++col )
            {
                camera.rotation[ row * 3 + col ] = rotation( row, col );
            }
            camera.translation[ row ] = images[ idx ]->m_pose_c2w_t( row );
        }
        image.data = images[ idx ]->m_img.data;
        image.row_stride = static_cast< int >( images[ idx ]->m_img.step );
        image.channels = images[ idx ]->m_img.channels();
    }
}

bool parse_frame_list( const std::string &frame_list_path, const std::string &dataset_root, size_t camera_count, bool use_frame_pose,
                       std::vector< OfflineFrameEntry > &frames )
{
    std::ifstream ifs( frame_list_path.c_str() );
    std::string   line;
    int           line_number = 0;
    if ( !ifs.is_open() )
    {
        cout << ANSI_COLOR_RED_BOLD << "Failed to open frame list: " << frame_list_path << ANSI_COLOR_RESET << endl;
        return false;
    }

    frames.clear();
    while ( std::getline( ifs, line ) )
    {
        std::stringstream       ss;
        std::vector< std::string > tokens;
        OfflineFrameEntry       frame;
        line_number++;
        line = trim_copy( line );
        if ( line.empty() || line[ 0 ] == '#' )
        {
            continue;
        }

        ss.str( line );
        while ( ss.good() )
        {
            std::string token;
            ss >> token;
            if ( token.empty() )
            {
                continue;
            }
            tokens.push_back( token );
        }

        const size_t expected_with_pose = 9 + camera_count;
        const size_t expected_without_pose = 2 + camera_count;
        if ( tokens.size() != expected_with_pose && ( use_frame_pose || tokens.size() != expected_without_pose ) )
        {
            cout << ANSI_COLOR_RED_BOLD << "Invalid frame list line " << line_number << ": expected "
                 << ( use_frame_pose ? std::to_string( expected_with_pose ) : std::to_string( expected_without_pose ) + " or " +
                                                                        std::to_string( expected_with_pose ) )
                 << " columns, got " << tokens.size() << ANSI_COLOR_RESET << endl;
            return false;
        }

        frame.timestamp = std::stod( tokens[ 0 ] );
        frame.frame_id = get_file_stem( tokens[ 1 ] );
        frame.pcd_path = resolve_path( dataset_root, tokens[ 1 ] );

        size_t image_token_offset = 2;
        if ( tokens.size() == expected_with_pose )
        {
            frame.lidar_t << std::stod( tokens[ 2 ] ), std::stod( tokens[ 3 ] ), std::stod( tokens[ 4 ] );
            frame.lidar_q = Eigen::Quaterniond( std::stod( tokens[ 5 ] ), std::stod( tokens[ 6 ] ), std::stod( tokens[ 7 ] ),
                                                std::stod( tokens[ 8 ] ) );
            frame.lidar_q.normalize();
            image_token_offset = 9;
        }
        for ( size_t idx = 0; idx < camera_count; ++idx )
        {
            frame.image_paths.push_back( resolve_path( dataset_root, tokens[ image_token_offset + idx ] ) );
        }
        frames.push_back( frame );
    }

    return !frames.empty();
}

bool build_frames_from_directory_layout( const std::string &dataset_root, const std::string &pose_dir, const std::string &pcd_dir,
                                         const std::vector< std::string > &camera_names, const std::string &image_extension,
                                         bool use_frame_pose,
                                         std::vector< OfflineFrameEntry > &frames )
{
    const std::string pcd_root = resolve_path( dataset_root, pcd_dir );
    const std::string pose_root = resolve_path( dataset_root, pose_dir );
    std::vector< std::string > pcd_files = list_files_with_suffix( pcd_root, ".pcd" );
    frames.clear();

    if ( pcd_files.empty() )
    {
        cout << ANSI_COLOR_RED_BOLD << "No .pcd files found in " << pcd_root << ANSI_COLOR_RESET << endl;
        return false;
    }

    for ( size_t idx = 0; idx < pcd_files.size(); ++idx )
    {
        OfflineFrameEntry frame;
        frame.frame_id = get_file_stem( pcd_files[ idx ] );
        frame.timestamp = static_cast< double >( idx );
        frame.pcd_path = pcd_files[ idx ];
        frame.pose_path.clear();
        if ( use_frame_pose == false )
        {
            frame.lidar_t.setZero();
            frame.lidar_q = Eigen::Quaterniond::Identity();
        }
        else
        {
            frame.pose_path = resolve_path( pose_root, frame.frame_id + ".txt" );
            if ( is_regular_file( frame.pose_path ) == false )
            {
                cout << ANSI_COLOR_RED_BOLD << "Missing pose file: " << frame.pose_path << ANSI_COLOR_RESET << endl;
                return false;
            }
            if ( load_pose_file( frame.pose_path, frame ) == false )
            {
                return false;
            }
        }

        for ( size_t camera_idx = 0; camera_idx < camera_names.size(); ++camera_idx )
        {
            const std::string image_path =
                resolve_path( dataset_root, camera_names[ camera_idx ] + "/" + frame.frame_id + image_extension );
            if ( is_regular_file( image_path ) == false )
            {
                cout << ANSI_COLOR_RED_BOLD << "Missing image file: " << image_path << ANSI_COLOR_RESET << endl;
                return false;
            }
            frame.image_paths.push_back( image_path );
        }
        frames.push_back( frame );
    }

    return !frames.empty();
}

void save_frame_cloud( const std::string &output_path, const std::vector< RGB_pt_ptr > &frame_points )
{
    pcl::PointCloud< pcl::PointXYZRGB > cloud_rgb;
    cloud_rgb.resize( frame_points.size() );
    for ( size_t idx = 0; idx < frame_points.size(); ++idx )
    {
        const vec_3 pos = frame_points[ idx ]->get_pos();
        const vec_3 rgb = frame_points[ idx ]->get_rgb();
        cloud_rgb.points[ idx ].x = pos( 0 );
        cloud_rgb.points[ idx ].y = pos( 1 );
        cloud_rgb.points[ idx ].z = pos( 2 );
        cloud_rgb.points[ idx ].b = static_cast< uint8_t >( std::max( 0.0, std::min( 255.0, rgb( 0 ) ) ) );
        cloud_rgb.points[ idx ].g = static_cast< uint8_t >( std::max( 0.0, std::min( 255.0, rgb( 1 ) ) ) );
        cloud_rgb.points[ idx ].r = static_cast< uint8_t >( std::max( 0.0, std::min( 255.0, rgb( 2 ) ) ) );
        cloud_rgb.points[ idx ].a = 255;
    }
    pcl::io::savePCDFileBinary( output_path, cloud_rgb );
}

int selection_mode_from_string( const std::string &mode )
{
    if ( mode == "first_valid" )
    {
        return C_COLORIZE_SELECT_FIRST_VALID;
    }
    return C_COLORIZE_SELECT_NEAREST_DISTANCE;
}

int sample_mode_from_string( const std::string &mode )
{
    if ( mode == "nearest" )
    {
        return C_COLORIZE_SAMPLE_NEAREST;
    }
    return C_COLORIZE_SAMPLE_BILINEAR;
}

OfflineStats colorize_frame_points( const std::vector< std::shared_ptr< Image_frame > > &images, std::vector< RGB_pt_ptr > &frame_points,
                                    std::vector< std::vector< RGB_pt_ptr > > *points_per_camera, double obs_time, int sample_mode,
                                    int selection_mode )
{
    OfflineStats                        stats( images.size() );
    std::vector< c_colorize_camera_t >  cameras;
    std::vector< c_colorize_image_t >   image_views;
    fill_colorize_inputs( images, cameras, image_views );

    for ( size_t idx = 0; idx < frame_points.size(); ++idx )
    {
        c_colorize_result_t result = {};
        double              world_point[ 3 ] = { frame_points[ idx ]->m_pos[ 0 ], frame_points[ idx ]->m_pos[ 1 ], frame_points[ idx ]->m_pos[ 2 ] };
        if ( c_colorize_select_point( cameras.data(), image_views.data(), static_cast< int >( cameras.size() ), world_point, sample_mode,
                                      selection_mode, &result ) == 0 )
        {
            stats.projection_fail++;
            if ( result.failure_reason == C_COLORIZE_FAIL_OUT_OF_BOUNDS )
            {
                stats.out_of_bounds++;
            }
            else if ( result.failure_reason == C_COLORIZE_FAIL_BEHIND_CAMERA )
            {
                stats.behind_camera++;
            }
            else
            {
                stats.invalid_input++;
            }
            continue;
        }

        stats.projection_success++;
        if ( result.camera_index >= 0 && static_cast< size_t >( result.camera_index ) < stats.per_camera_hits.size() )
        {
            stats.per_camera_hits[ result.camera_index ]++;
        }

        if ( points_per_camera != nullptr && result.camera_index >= 0 &&
             static_cast< size_t >( result.camera_index ) < points_per_camera->size() )
        {
            ( *points_per_camera )[ result.camera_index ].push_back( frame_points[ idx ] );
        }

        double u = result.u;
        double v = result.v;
        const double gray = images[ result.camera_index ]->get_grey_color( u, v, 0 );
        frame_points[ idx ]->update_gray( gray, result.camera_distance );
        stats.rgb_updates += frame_points[ idx ]->update_rgb(
            vec_3( result.bgr[ 0 ], result.bgr[ 1 ], result.bgr[ 2 ] ), result.camera_distance,
            vec_3( kOfflineImageObsCov, kOfflineImageObsCov, kOfflineImageObsCov ), obs_time );
    }

    return stats;
}

void log_stats( const OfflineStats &stats, size_t frame_idx, const std::vector< OfflineCameraConfig > &cameras )
{
    std::ostringstream oss;
    oss << "[offline_colorize] frame=" << frame_idx << " proj_ok=" << stats.projection_success << " proj_fail=" << stats.projection_fail
        << " oob=" << stats.out_of_bounds << " behind=" << stats.behind_camera << " invalid=" << stats.invalid_input
        << " rgb_updates=" << stats.rgb_updates << " per_camera=[";
    for ( size_t idx = 0; idx < stats.per_camera_hits.size(); ++idx )
    {
        if ( idx )
        {
            oss << ",";
        }
        oss << cameras[ idx ].name << ":" << stats.per_camera_hits[ idx ];
    }
    oss << "]";
    scope_color( ANSI_COLOR_CYAN_BOLD );
    cout << oss.str() << ANSI_COLOR_RESET << endl;
}
} // namespace

int main( int argc, char **argv )
{
    OfflineAppConfig config;
    std::string      config_path = std::string( ROOT_DIR ).append( "../config/offline_colorize_config.yaml" );
    if ( argc > 1 )
    {
        const std::string arg1 = argv[ 1 ];
        if ( arg1 == "--help" || arg1 == "-h" )
        {
            print_usage( argv[ 0 ] );
            return 0;
        }
        config_path = argv[ 1 ];
    }
    if ( load_app_config( config_path, config ) == false )
    {
        return -1;
    }
    cv::FileStorage config_fs( config.config_path, cv::FileStorage::READ );
    if ( config_fs.isOpened() == false )
    {
        cout << ANSI_COLOR_RED_BOLD << "Failed to reopen config file: " << config.config_path << ANSI_COLOR_RESET << endl;
        return -1;
    }
    const cv::FileNode offline_node = config_fs[ "offline_colorize" ];
    if ( argc > 2 )
    {
        config.dataset_root = argv[ 2 ];
    }

    config.frame_list = resolve_path( config.dataset_root, config.frame_list );
    Common_tools::create_dir( config.output_dir );
    Common_tools::create_dir( config.output_dir + "/frames" );

    if ( config.camera_names.empty() )
    {
        cout << ANSI_COLOR_RED_BOLD << "offline_colorize/camera_names is empty." << ANSI_COLOR_RESET << endl;
        return -1;
    }

    std::vector< OfflineCameraConfig > cameras( config.camera_names.size() );
    for ( size_t idx = 0; idx < config.camera_names.size(); ++idx )
    {
        if ( load_camera_config( offline_node, config.camera_names[ idx ], cameras[ idx ] ) == false )
        {
            return -1;
        }
    }

    std::vector< OfflineFrameEntry > frames;
    if ( config.input_mode == "directory_layout" )
    {
        if ( build_frames_from_directory_layout( config.dataset_root, config.pose_dir, config.pcd_dir, config.camera_names,
                                             config.image_extension, config.use_frame_pose != 0, frames ) == false )
        {
            cout << ANSI_COLOR_RED_BOLD << "Failed to build frames from directory layout under " << config.dataset_root << ANSI_COLOR_RESET
                 << endl;
            return -1;
        }
    }
    else if ( parse_frame_list( config.frame_list, config.dataset_root, cameras.size(), config.use_frame_pose != 0, frames ) == false )
    {
        cout << ANSI_COLOR_RED_BOLD << "Failed to parse frame list: " << config.frame_list << ANSI_COLOR_RESET << endl;
        return -1;
    }

    Global_map            global_map( 0 );
    Offline_map_recorder  recorder;
    OfflineStats          total_stats( cameras.size() );
    const int             selection_mode = selection_mode_from_string( config.camera_selection_mode );
    const int             sample_mode = sample_mode_from_string( config.sample_mode );
    recorder.m_global_map = &global_map;
    recorder.set_working_dir( config.output_dir );
    global_map.set_minmum_dis( config.minimum_pts_size );

    scope_color( ANSI_COLOR_GREEN_BOLD );
    cout << "Offline colorize frames: " << frames.size() << ", cameras: " << cameras.size() << ANSI_COLOR_RESET << endl;
    if ( config.use_frame_pose == 0 )
    {
        scope_color( ANSI_COLOR_YELLOW_BOLD );
        cout << "Offline colorize is running without frame pose files; LiDAR pose defaults to identity for every frame."
             << ANSI_COLOR_RESET << endl;
        if ( frames.size() > 1 )
        {
            cout << ANSI_COLOR_YELLOW_BOLD
                 << "Multiple frames detected in no-pose mode; outputs are only meaningful when frames already share one coordinate system."
                 << ANSI_COLOR_RESET << endl;
        }
    }

    for ( size_t frame_idx = 0; frame_idx < frames.size(); ++frame_idx )
    {
        pcl::PointCloud< pcl::PointXYZI >              frame_cloud_lidar;
        pcl::PointCloud< pcl::PointXYZI >              frame_cloud_world;
        std::vector< std::shared_ptr< Image_frame > > images;
        std::vector< RGB_pt_ptr >                      frame_points;
        std::vector< std::vector< RGB_pt_ptr > >       points_per_camera( cameras.size() );

        if ( load_point_cloud_xyzi( frames[ frame_idx ].pcd_path, frame_cloud_lidar ) == false )
        {
            cout << ANSI_COLOR_RED_BOLD << "Failed to load pcd: " << frames[ frame_idx ].pcd_path << ANSI_COLOR_RESET << endl;
            return -1;
        }

        if ( config.frame_point_in_world )
        {
            frame_cloud_world = frame_cloud_lidar;
        }
        else
        {
            transform_cloud_to_world( frame_cloud_lidar, frames[ frame_idx ].lidar_q, frames[ frame_idx ].lidar_t, frame_cloud_world );
        }

        global_map.append_points_to_global_map( frame_cloud_world, frames[ frame_idx ].timestamp, &frame_points,
                                                config.append_global_map_point_step );

        for ( size_t camera_idx = 0; camera_idx < cameras.size(); ++camera_idx )
        {
            std::shared_ptr< Image_frame > image;
            if ( load_frame_image( cameras[ camera_idx ], frames[ frame_idx ], frames[ frame_idx ].image_paths[ camera_idx ], image ) == false )
            {
                cout << ANSI_COLOR_RED_BOLD << "Failed to load image: " << frames[ frame_idx ].image_paths[ camera_idx ] << ANSI_COLOR_RESET
                     << endl;
                return -1;
            }
            images.push_back( image );
        }

        OfflineStats frame_stats =
            colorize_frame_points( images, frame_points, config.save_offline_map ? &points_per_camera : nullptr,
                                   frames[ frame_idx ].timestamp, sample_mode, selection_mode );
        total_stats.merge( frame_stats );
        log_stats( frame_stats, frame_idx, cameras );

        if ( config.save_offline_map )
        {
            for ( size_t camera_idx = 0; camera_idx < images.size(); ++camera_idx )
            {
                recorder.insert_image_and_pts( images[ camera_idx ], points_per_camera[ camera_idx ] );
            }
        }

        if ( config.save_frame_colored_pcd )
        {
            std::ostringstream oss;
            oss << config.output_dir << "/frames/frame_" << std::setw( 6 ) << std::setfill( '0' ) << frame_idx << "_rgb.pcd";
            save_frame_cloud( oss.str(), frame_points );
        }
    }

    global_map.save_to_pcd( config.output_dir, "/rgb_map", 1 );
    if ( config.save_offline_map )
    {
        recorder.export_to_mvs( global_map );
    }

    log_stats( total_stats, frames.size(), cameras );
    return 0;
}
