# path = "data/wiki/"
path = "data/yago/"

YEARMIN = -50
YEARMAX = 3000

quadSet = set()
time_dict = {}


def preprocess(data_part):
    data_path = path+data_part+".txt"
    new_data_path = path+data_part+"2id.txt"
    fw = open(new_data_path, "w")
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            fw.write("%s\t%s\t%s\t%d\t%d\n" % (info[0], info[1], info[2], int(info[3]) - 1, 0))
    fw.close()


# stat()
preprocess("train")
preprocess("valid")
preprocess("test")
