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
    sub_id = 2
    ob_id = 8
    rel_id = 5
    time_id = 1



    # with open("dataset/events.2017.20180710093300.tsv") as fp:
    with open("dataset/events.2018.20181119132436.tsv") as fp:
        for i,line in enumerate(fp):
            # skip first line
            count += 1
            if count == 1:
                continue
            info = line.strip().split("\t")
        
            if info[sub_id] == info[ob_id]:
                continue

            entity1_id = None
            if info[sub_id] in entity_dict:
                entity1_id = entity_dict[info[sub_id]]
            else:
                entity1_id = len(entity_dict)
                entity_dict[info[sub_id]] = entity1_id

            entity2_id = None
            if info[ob_id] in entity_dict:
                entity2_id = entity_dict[info[ob_id]]
            else:
                entity2_id = len(entity_dict)
                entity_dict[info[ob_id]] = entity2_id

            relation_id = None
            if info[rel_id] in relation_dict:
                relation_id = relation_dict[info[rel_id]]
            else:
                relation_id = len(relation_dict)
                relation_dict[info[rel_id]] = relation_id

            delta = datetime.datetime.strptime(info[time_id], format) - datetime.datetime.strptime(start_time, format)
            timestamp = delta.days * 24
            
            if timestamp < 4344:
                fw1.write("%-5d\t%-5d\t%-3d\t%-3d\t0\n" % (entity1_id, relation_id, entity2_id, timestamp))
            else:
                fw2.write("%-5d\t%-5d\t%-3d\t%-3d\t0\n" % (entity1_id, relation_id, entity2_id, timestamp))

    print(count)

    fw3.write(str(len(entity_dict)) + "\t" + str(len(relation_dict))+"\t0")
    fw1.close()
    fw2.close()
    fw3.close()

if __name__ == "__main__":
    preprocess()