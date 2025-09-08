from dataset_related.detect_mesh_and_keypoint import DetectMeshandKeypoint_1


dmk=DetectMeshandKeypoint_1()

video_file_path="/remote-home/share/yfsong/shipu/video_src/part_video/video_27/segments/video_27_high_quanlity_clip_100.mp4"
out_dir="/remote-home/share/yfsong/shipu/10.24"

dmk.get_mesh_and_keypoint_pic(video_file_path,out_dir)

"""
我们主要是看一下smirk的编码器的输出是什么

dict_keys(['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params'])

"""
