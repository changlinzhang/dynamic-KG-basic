import datetime


def preprocess():
    entity_dict = {}
    relation_dict = {}

    # fw1 = open("dataset/events.2017.20180710093300/train.txt", "w")
    # fw2 = open("dataset/events.2017.20180710093300/test.txt", "w")
    # fw3 = open("dataset/events.2017.20180710093300/stat.txt", "w")
    # start_time = "2017-01-01"

    fw1 = open("dataset/events.2018.20181119132436/train.txt", "w")
    fw2 = open("dataset/events.2018.20181119132436/test.txt", "w")
    fw3 = open("dataset/events.2018.20181119132436/stat.txt", "w")
    start_time = "2018-01-01"

    count = 0
    train_test_split = 700000
    format = "%Y-%m-%d"



    # with open("dataset/events.2017.20180710093300.tsv") as fp:
    with open("dataset/events.2018.20181119132436.tsv") as fp:
        for i,line in enumerate(fp):
            # skip first line
            count += 1
            if count == 1:
                continue
            info = line.strip().split("\t")

            entity1_id = None
            if info[2] in entity_dict:
                entity1_id = entity_dict[info[2]]
            else:
                entity1_id = len(entity_dict)
                entity_dict[info[2]] = entity1_id

            entity2_id = None
            if info[8] in entity_dict:
                entity2_id = entity_dict[info[8]]
            else:
                entity2_id = len(entity_dict)
                entity_dict[info[8]] = entity2_id

            relation_id = None
            if info[5] in relation_dict:
                relation_id = relation_dict[info[5]]
            else:
                relation_id = len(relation_dict)
                relation_dict[info[5]] = relation_id

            delta = datetime.datetime.strptime(info[1], format) - datetime.datetime.strptime(start_time, format)
            timestamp = delta.days

            fw1.write("%-5d %-5d %-3d %-3d\n" % (entity1_id, entity2_id, relation_id, timestamp))

    print(count)
    fw3.write(str(len(entity_dict)) + " " + str(len(relation_dict)))
    fw1.close()
    fw2.close()
    fw3.close()

if __name__ == "__main__":
    preprocess()

