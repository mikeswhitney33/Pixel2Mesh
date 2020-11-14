import pickle
from p2m_utils.render_utils import PCRenderer
from p2m_utils import rough_model
from p2m_utils.dat_utils import obj2dat
from p2m_utils import path_utils
import os

def pcfile2datfile(pc_file, out_file):
    dat = pcfile2dat(pc_file)
    with open(out_file, "wb") as ofile:
        pickle.dump(dat, ofile, protocol=2)

def pcfile2dat(pc_file):
    renderer = PCRenderer()
    front = renderer.render(pc_file, front_side="front")
    side = renderer.render(pc_file, front_side="side")

    raw_mesh = rough_model.make_rough_model(front, side)
    dat = obj2dat(raw_mesh)
    return dat



def transform_dataset(data_list):
    with open(data_list, "r") as ifile:
        file_names = map(lambda x : x.strip(), ifile.readlines())
    for pcfile in file_names:
        outfile = pcfile.replace("rendering", "features")
        feature_dir = os.path.dirname(outfile)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        pcfile2datfile(pcfile, outfile)


if __name__ == "__main__":
    transform_dataset(os.path.join(path_utils.get_data_dir(), "train_list.txt"))
