#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// AirVO
#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"

// #include "rcf.h"
// #include "mutex"

MapBuilder* p_map_builder;

std::mutex _rcf_mutex;


void GrabStereo(const sensor_msgs::ImageConstPtr& imgLeft, const sensor_msgs::ImageConstPtr& imgRight){
    cv_bridge::CvImageConstPtr cv_ptrLeft, cv_ptrRight; //-------改

    try{
        cv_ptrLeft = cv_bridge::toCvShare(imgLeft, sensor_msgs::image_encodings::MONO8);
        cv_ptrRight = cv_bridge::toCvShare(imgRight, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    static int frame_id = 0;

    InputDataPtr input_data = std::shared_ptr<InputData>(new InputData());


    input_data->index = frame_id;
    input_data->image_left = cv_ptrLeft->image.clone();
    input_data->image_right = cv_ptrRight->image.clone();
    input_data->time = imgLeft->header.stamp.toSec();



    if(input_data == nullptr) return;
    p_map_builder->AddInput(input_data);
    std::cout << "ROS get frame id is == " << frame_id++ << std::endl; // 这个id就没有意义
    // frame_id++;

}

int main(int argc, char **argv){
    ros::init(argc, argv, "air_vo_ros");

    // AirVO
    std::string config_path, model_dir;
    ros::param::get("~config_path", config_path);
    ros::param::get("~model_dir", model_dir);
    Configs configs(config_path, model_dir);
    ros::param::get("~camera_config_path", configs.camera_config_path);
    ros::param::get("~saving_dir", configs.saving_dir);
    std::string traj_path;
    ros::param::get("~traj_path", traj_path);

    p_map_builder = new MapBuilder(configs);//这里为什么不用智能指针申请对象，也没delete。

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1){
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }

    // ROS
    std::string left_topic, right_topic,left_topic_rcf,right_topic_rcf; //-------改
    ros::param::get("~left_topic", left_topic);
    ros::param::get("~right_topic", right_topic);


    ros::NodeHandle node_handler;
    message_filters::Subscriber<sensor_msgs::Image> sub_img_left(node_handler, left_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> sub_img_right(node_handler, right_topic, 1);


    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), sub_img_left, sub_img_right); //-------改
    sync.registerCallback(boost::bind(&GrabStereo, _1, _2)); //-------改



    // Starts the operation
    ros::spin();

    // Shutting down
    std::cout << "Saving trajectory to " << traj_path << std::endl;
    p_map_builder->SaveTrajectory(traj_path);

    ros::shutdown();

    std::cout << "map size is: " << p_map_builder->_map->_mappoints.size() << std::endl;
    

    return 0;
}