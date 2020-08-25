from collections import Counter
from docx import Document
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold     
import pandas as pd, sys, pickle, matplotlib.pyplot as plt, numpy as np, os, gc
from words_interest_log_reg import words_interest_log_reg, create_clf
from vectorizer_texts_tfidf import vectorizer_texts_tfidf

if __name__ == "__main__":
    # ****** PROCESSING OF TEXTS FROM ALL CLASSES ******
    # REGRESSÃO LOGÍSTICA COM LASSO PARA ESCOLHER AS PALAVRAS MAIS CARACTERÍSTICAS DOS TEXTOS
    # CSV PATH. O input esperado é um csv com n colunas correspondendo às variáveis de regex
    # e uma colunas chamada: 'class'
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    classes_target = Counter(df['class'].tolist())
    classes = df['class'].tolist()
    df.drop(['class'], axis=1, inplace=True)
    X = df.to_numpy()
    for target in classes_target:
        y = []
        for i in classes:
            if i == target:
                y.append(1)
            else:
                y.append(0)
        cv = RepeatedStratifiedKFold(random_state=1)
        # accuracy
        scores = cross_val_score(LogisticRegression(penalty='l1',n_jobs=-1,solver='saga'), X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        plt.boxplot(scores,showmeans=True)
        plt.savefig('Acurácia do modelo RL para a classe {}.png'.format(target))
        plt.clf()
        # recall or sensitivity
        scores = cross_val_score(LogisticRegression(penalty='l1',n_jobs=-1,solver='saga'), X, y, scoring='recall', cv=cv, n_jobs=-1, error_score='raise')
        plt.boxplot(scores,showmeans=True)
        plt.savefig('Sensibilidade do modelo RL para a classe {}.png'.format(target))
        plt.clf()
        # precision
        scores = cross_val_score(LogisticRegression(penalty='l1',n_jobs=-1,solver='saga'), X, y, scoring='precision', cv=cv, n_jobs=-1, error_score='raise')
        plt.boxplot(scores,showmeans=True)
        plt.savefig('Precisão do modelo RL para a classe {}.png'.format(target))
        plt.clf()
        gc.collect()
        classifier = create_clf(X,y)
        variables_of_interest = words_interest_log_reg(classifier,df.columns)
        rows = []
        for beta, variable in variables_of_interest:
            rows.append({'beta':beta,'variable':variable})
        df_target = pd.DataFrame(rows,index=[i for i in range(len(rows))])
        df_target.to_csv('betas_regressão_classe_'+str(target)+'.csv',index=False)
    


