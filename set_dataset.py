import os
import shutil

rootDir = "/home/sba/guoyanjun/dhrnet"
classDirs = ["models", "plants", "portraits"]
trainDirs = [ os.path.join(os.path.join(rootDir, d),d+"_train") for d in classDirs]
testDirs = [os.path.join(os.path.join(rootDir, d), d + "_test") for d in classDirs]

def getClassFileDir(dirs,mod = "train"):
    ret = []
    ret_ = {}
    for d in dirs:
        imgdirs = os.listdir(d);
        for imgdir in imgdirs:
            ret.append(os.path.join(d, imgdir, imgdir + "_HR_gt.png"))
            ret.append(os.path.join(d, imgdir, imgdir + "_LR_fill_depth.png"))
            ret.append(os.path.join(d, imgdir, imgdir + "_RGB.jpg"))
            ret_[os.path.join(d, imgdir, imgdir + "_HR_gt.png")] = os.path.join(rootDir, mod, "HR_GT", imgdir + ".png")
            ret_[os.path.join(d, imgdir, imgdir + "_LR_fill_depth.png")] = os.path.join(rootDir, mod, "LR_DEPTH", imgdir + ".png")
            ret_[os.path.join(d, imgdir, imgdir + "_RGB.jpg")] = os.path.join(rootDir, mod, "RGB",imgdir + ".jpg")
    return ret, ret_
def moveFile(oris, dsts):
    for ori in oris:
        shutil.copyfile(ori,dsts[ori])
ret,ret_ = getClassFileDir(trainDirs)
moveFile(ret, ret_)
ret, ret_ = getClassFileDir(testDirs,"test")
moveFile(ret, ret_)
