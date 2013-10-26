import string


trigram = list()
count_no = list()
text_file = open('test_data.txt', 'r')
myList = (''.join(char if char not in string.punctuation else ' ' +
                  char for char in text_file.read())).split(' ')
temp1 = ''
temp2 = '\n'
for temp_word in myList:
        temp_list = [temp1, temp2, temp_word,1]
        if trigram.count(temp_list) is 0:
                trigram.append(temp_list)
                count_no.append(1)
        else:
                count_no[trigram.index(temp_list)] += 1
        temp1 = temp2
        temp2 = temp_word
work = open('trigram.txt', 'w')
work.writelines(["%s\n" % item for item in trigram])
work1 = open('count.txt', 'w')
work1.writelines(["%s\n" % [item,value] for item,value in enumerate(count_no)])
work.close()
work1.close()
text_file.close()
