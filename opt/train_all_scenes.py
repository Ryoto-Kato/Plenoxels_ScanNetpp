import sys, os
from argparse import ArgumentParser
from util.dataset_handler import Filehandler

if __name__=="__main__":

    # path2data="/home/kato/Documents/HiWi/spp_benchmark/samples"
    path2data="/home/kato/Documents/HiWi/spp_benchmark/nvstest_benchmarkdata_v2-undistorted_fixcenter"
    
    for i in range(1):
        config_id = 505+i

        parser = ArgumentParser(description="train for all scenes")
        parser.add_argument('--train_dir', type = str, default=None) #./ckpt/name of scene + "scale0.5_allframes_json26"
        parser.add_argument('--data_dir', type = str, default= None)
        parser.add_argument('--scale', type = float, default=0.5)
        parser.add_argument('--config', type = str, default=f"./configs/sample{config_id}.json")
        parser.add_argument('--gpu', type = int, default=0)
        parser.add_argument('--dataformat', type =str, default="nsvf")
        _args = parser.parse_args(sys.argv[1:])

        path2ckpt_dir = os.path.join(os.getcwd(),"ckpt", f"json{config_id}")
        list_scene_name, list_scene_path = Filehandler.dirwalker_InFolder(path_to_folder=path2data, prefix = "")

        for scene_name, scene_path in zip(list_scene_name, list_scene_path):   
            print(f"{'==='*3} current training scene name: {scene_name}{'==='*3}")
            CKPT_DIR = os.path.join(path2ckpt_dir, scene_name+f"scale{_args.scale}_allframes_json{config_id}")
            print(f"CKPT_DIR: {CKPT_DIR}")
            os.makedirs(CKPT_DIR, exist_ok=True)
            # NSVF *normalized before hand
            if _args.dataformat=="nsvf":
                DATA_DIR = os.path.join(scene_path, "dslr", "nsvf")
            else:
                # transforms.json
                print("nerfstudio")
                DATA_DIR = os.path.join(scene_path, "dslr", "nerfstudio")
            with open(os.path.join(DATA_DIR, "readme.txt"), 'r') as readme_f:
                _context = readme_f.read()
            print(_context)
            NOHUP_FILE = os.path.join(CKPT_DIR, "log")
            print(f"NOHUP_FILE: {NOHUP_FILE}")

            os.system(f"CUDA_VISIBLE_DEVICES=0 nohup python -u opt.py -t {CKPT_DIR} {DATA_DIR} -c {_args.config} --scene_name {scene_name} --scale {_args.scale} > {NOHUP_FILE} 2>&1") 
