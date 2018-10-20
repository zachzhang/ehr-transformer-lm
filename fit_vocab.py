import pandas as pd
import util2 as u2
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def load_data():
    X = pd.read_csv("/ifs/data/razavianlab/ehr_training_data/master_dataset.csv.gz",compression='gzip',chunksize = 20000)

    df = []
    
    for x in X:    
        df.append(x)
    
    return pd.concat(df, axis= 0)


if __name__ == '__main__':


    X = load_data()

    X['NOTE_TEXT'] = X['NOTE_TEXT'].apply(u2.cleanNotes)

    vect = CountVectorizer(max_features=40000)
    vect.fit(X['NOTE_TEXT'])
    pickle.dump(vect, open('vect.p','wb'))


    
