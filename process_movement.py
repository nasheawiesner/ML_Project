import numpy as np
def parse():
    training_data = []
    labels = []
    with open("subject7.txt", encoding='utf-8') as input_file:
        for line in input_file:
            cols = line.split(',')
            example = []
            for feature in cols:
                if '\\\n' not in feature:
                        if feature != '0\\':
                            if feature != '':
                                if feature != '\n':
                                    example.append(float(feature))
                else:
                    labels.append(float(feature[0]))
            training_data.append(example)
    input_file.close()
    training_data = np.array(training_data)
    labels = np.asarray(labels)
    return training_data[0:10000], labels



