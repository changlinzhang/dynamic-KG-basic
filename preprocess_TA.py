# wiki
# valid begin at 72 (2008, 2008)
# test begin at 77 (2013, 2013)
# yago
# valid begin at 65 (2011, 2011)
# test begin at 68 (2014, 2014)

import functools

path = "data/wiki_original/"
new_path = "data/wiki/"
valid_begin_year = 2008
test_begin_year = 2013
# path = "data/yago_original/"
# new_path = "data/yago/"
# valid_begin_year = 2011
# test_begin_year = 2014

quadList = []
time_dict = {}

def stat():
    fw = open(new_path + "stat.txt", "w")

    epath = path + "entity2id.txt"
    ecount = 0
    with open(epath) as fp:
        for i,line in enumerate(fp):
            ecount += 1

    rcount = 0
    rpath = path + "relation2id.txt"
    with open(rpath) as fp:
        for i,line in enumerate(fp):
            rcount += 1

    fw.write("%d\t%d\t%d\n" % (ecount, rcount, int(quadList[-1][3]) + 1))
    fw.close()


def preprocess(data_part):
    data_path = path+data_part+".txt"
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            start_year = info[3].split('-')[0]
            end_year = info[4].split('-')[0]
            if start_year and start_year.find('#') == -1:
                quadList.append((info[0], info[1], info[2], 0, int(start_year)))  # 0 for since
            if end_year and end_year.find('#') == -1:
                quadList.append((info[0], info[1], info[2], 1, int(end_year)))  # 1 for until
                # fw.write("%s\t%s\t%s\t%d\t%d\n" % (info[0], info[1], info[2], year, 0))


def save():
    new_train_path = new_path + "train2id.txt"
    new_valid_path = new_path + "valid2id.txt"
    new_test_path = new_path + "test2id.txt"

    # quadList = list(quadSet)

    # def func(x, y):
    #     if x[4] < y[4]:
    #         return -1
    #     if x[4] > y[4]:
    #         return 1
    #     else:
    #         if x[3] < y[3]:
    #             return -1
    #         else:
    #             return 1

    list.sort(quadList, key=lambda quad: quad[4])
    # sorted(quadList, key=functools.cmp_to_key(func))
    print(quadList)
    # list.sort(quadList, key=lambda quad: quad[3]) # sorted by timestamp
    total = len(quadList)

    index = 0
    fw = open(new_train_path, "w")
    while True:
        quad = quadList[index]
        index += 1
        if quad[4] >= valid_begin_year:
            break
        fw.write("%s\t%s\t%s\t%d\t%04d\n" % (quad[0], quad[1], quad[2], quad[3], quad[4]))
    fw.close()

    fw = open(new_valid_path, "w")
    while True:
        quad = quadList[index]
        index += 1
        if quad[4] >= test_begin_year:
            break
        fw.write("%s\t%s\t%s\t%d\t%04d\n" % (quad[0], quad[1], quad[2], quad[3], quad[4]))
    fw.close()

    fw = open(new_test_path, "w")
    while index < total:
        quad = quadList[index]
        index += 1
        fw.write("%s\t%s\t%s\t%d\t%04d\n" % (quad[0], quad[1], quad[2], quad[3], quad[4]))
    fw.close()


# preprocess("triple2id")
preprocess("train")
preprocess("valid")
preprocess("test")
save()
