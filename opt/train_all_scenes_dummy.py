import sys, os
from argparse import ArgumentParser
from util.dataset_handler import Filehandler

if __name__=="__main__":

    path2data="/home/kato/Documents/HiWi/spp_benchmark/nvstest_benchmarkdata_v2-undistorted_fixcenter"

    parser = ArgumentParser(description="train for all scenes")
    parser.add_argument('--train_dir', type = str, default=None) #./ckpt/name of scene + "scale0.5_allframes_json26"
    parser.add_argument('--data_dir', type = str, default= None)
    parser.add_argument('--scale', type = float, default=0.25)
    parser.add_argument('--config', type = str, default="./configs/sample26.json")
    parser.add_argument('--gpu', type = int, default=0)
    _args = parser.parse_args(sys.argv[1:])

    list_scene_name, list_scene_path = Filehandler.dirwalker_InFolder(path_to_folder=path2data, prefix = "")

    for scene_name, scene_path in zip(list_scene_name, list_scene_path):   
        print(f"{'==='*3} current training scene name: {scene_name}{'==='*3}")
        CKPT_DIR = os.path.join(os.getcwd(),"ckpt", scene_name+f"scale_{_args.scale}_allframes_json{_args.config}")
        print(f"CKPT_DIR: {CKPT_DIR}")
        DATA_DIR = os.path.join(scene_path, "dslr", "nsvf")
        with open(os.path.join(DATA_DIR, "readme.txt"), 'r') as readme_f:
            _context = readme_f.read()
        print(_context)
        NOHUP_FILE = os.path.join(CKPT_DIR, "log")
        print(f"NOHUP_FILE: {NOHUP_FILE}")
    