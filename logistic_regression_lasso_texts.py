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
    # CSV PATH. O input esperado é um csv com duas colunas: 'text', 'class'
    csv_path = sys.argv[1]
    # vetorização de textos e classes como class
    df = pd.read_csv(csv_path)
    classes_target = Counter(df['class'].tolist())
    vectorizer = vectorizer_texts_tfidf(csv_path, use_pca=False)
    var_names = vectorizer.vectorizer.get_feature_names()
    X = vectorizer.X
    for target in classes_target:
        y = []
        for i in vectorizer.y:
            if i == target:
                y.append(1)
            else:
                y.append(0)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
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
        variables_of_interest = words_interest_log_reg(classifier,var_names)
        rows = []
        for beta, word in variables_of_interest:
            rows.append({'beta':beta,'word':word})
        df_target = pd.DataFrame(rows,index=[i for i in range(len(rows))])
        df_target.to_csv('betas_regressão_classe_'+str(target)+'.csv',index=False)
    


