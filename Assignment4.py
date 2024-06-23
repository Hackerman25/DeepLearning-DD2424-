
def getdata():
    fileurl = r"C:\Users\marcu\Documents\GitHub\DeepLearning-DD2424-\Datasets\goblet_book.txt"
    file = open(fileurl, "r")


    return file.read()


def findnumbunique(file):

    output = len(set(file))

    return output

def createmappingfunc(file):

    output1 = sorted(set(file))

    my_map = dict((j,i) for i,j in enumerate(output1))
    inv_map = {v: k for k, v in my_map.items()}

    return my_map, inv_map




if __name__ == "__main__":

    data = getdata()


    numbunique = findnumbunique(data)
    print(numbunique)



    my_map, inv_map = createmappingfunc(data)
    print(my_map)
    print(inv_map)


    print(my_map[data[0]])
    print(inv_map[0])