from termcolor import colored
def reg_met(func):
    """{y_actual = None,y_predict = None, mse = True, mae = True, rmse = True, r_square = True, sample_weight = None,  multioutput = uniform_average, squared = True}"""
    def inner(**kwargs):

        default = {'y_actual': None,'y_predict':None, 'mse':True, 'mae':True, 'rmse': True, 'r_square' : True,
        'sample_weight' : None, 'multioutput':'uniform_average', 'squared':True}
        default.update(kwargs)

        from IPython.display import display
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import math
        import pandas as pd
        li1 = []
        li2 = []
        if(default['mse']):
            li1.append("MSE")
            li2.append(mean_squared_error(kwargs['y_actual'], kwargs['y_predict']))
        if(default['mae']):
            li1.append("MAE")
            li2.append(mean_absolute_error(kwargs['y_actual'], kwargs['y_predict']))
        if(default['rmse']):
            li1.append("RMSE")
            li2.append(math.sqrt(mean_squared_error(kwargs['y_actual'], kwargs['y_predict'])))
        if(default['r_square']):
            li1.append("R square")
            li2.append(r2_score(kwargs['y_actual'], kwargs['y_predict']))

        data_tuples = list(zip(li1,li2))
        df = pd.DataFrame(data_tuples, columns=['Metrics','Score'])
        display(df)
    return inner

def text_cleaning(func):
    """{clean_stopwords = False, clean_punc = False, clean_unwanted_char = False, 
                  to_lowercase = False, Stemming = False, lemmatization = False, return_sent= False}"""
    def process(filename,**kwargs):
        default ={'clean_stopwords': False, 'clean_punc': False, 'clean_unwanted_char': False, 
                  'to_lowercase': False, 'Stemming': False, 'lemmatization': False, 'return_sent': False}
        default.update(kwargs)
        import nltk
        import re
        import string
        import pandas as pd
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        from nltk.stem import WordNetLemmatizer
        from IPython.display import display
        nltk.download('stopwords',quiet=True)
        nltk.download('wordnet',quiet=True)
        corpus = []
        df=func(filename)
        if(default['clean_unwanted_char']==True):
            df=df.apply(lambda x:re.sub(pattern='[^a-zA-Z]', repl=' ', string=x))
        if(default['clean_punc']==True):
            df=df.apply(apply(lambda x: x.replace(c,'') for c in string.punctuation))
        if(default['to_lowercase']==True):
            df=df.apply(lambda x:x.lower())
        df=df.apply(lambda x:x.split())
        if(default['clean_stopwords']==True):
            df = df.apply(lambda x:[word for word in x if word not in set(stopwords.words('english'))])
        if(default['Stemming']==True):
            ps = PorterStemmer()
            df = df.apply(lambda x:[ps.stem(word) for word in x])
        if(default['lemmatization']==True):
            lm = WordNetLemmatizer()
            df = df.apply(lambda x:[lm.lemmatize(word) for word in x])
        if(default['return_sent']==True):
            df = df.apply(lambda x:' '.join(x))
        df = df.values.tolist()
        data = pd.DataFrame(df, columns=['text_cleaning'])
        return data 
    return process


def classification_met(func):
    """{y_actual = None, y_predict=None, classifier = None, y_prob = None, accuracy_score = True,confusion_matrix = True, 
        classification_report  = True, confusionmatrixplot  = True, a_normalize = True, a_sample_weight = None,
        cr_labels = None, cr_target_names = None, cr_sample_weight = None, digits = 2, output_dict = False,
        zero_division = 'warn', cm_labels = None, cm_sample_weight = None, normalize = None,data = None, display_labels = None,
        average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None} """
    def process(**kwargs):
        #display(kwargs)
        default = {'y_actual' : None, 'y_predict':None,'accuracy_score': True,
       'confusion_matrix': True,'classification_report' : True, 'confusionmatrixplot' : True, 'classifier' : None, 'x_test' : None,       
        'a_normalize':True, 'a_sample_weight':None,'cr_labels':None, 'cr_target_names':None, 'cr_sample_weight':None, 'digits':2, 'output_dict':False, 
        'zero_division':'warn','cm_labels':None, 'cm_sample_weight':None, 'normalize':None,'data':None,'display_labels':None,
        'average':'macro', 'sample_weight':None, 'max_fpr':None, 'multi_class':'raise', 'labels':None}
        default.update(kwargs)
        
        from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score
        from sklearn.metrics import plot_confusion_matrix
        from IPython.display import display
        from sklearn.metrics import roc_curve
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as plt
        import numpy as np
        if(default['accuracy_score']):
            print(colored("Accuracy Score: ",'blue'),accuracy_score(default['y_actual'], default['y_predict'],normalize=default['a_normalize'], sample_weight=default['a_sample_weight']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['classification_report']):
            print(colored("Classification report:\n",'blue'),classification_report(default['y_actual'], default['y_predict'], labels=default['cr_labels'], 
                                                               target_names=default['cr_target_names'],sample_weight=default['cr_sample_weight'], digits=default['digits'], output_dict=default['output_dict'], 
                                                               zero_division=default['zero_division']))
            print(colored("----------------------------------------------------------",'red'))
                                                                                                                                  
        if(default['confusion_matrix']):                                                                                                                          
            matrix = confusion_matrix(default['y_actual'], default['y_predict'],
                                 labels=default['cm_labels'], sample_weight=default['cm_sample_weight'], normalize=default['normalize'])
            print(colored("Confusion Matrix:\n",'blue'),matrix)
            
        
        if(default['confusionmatrixplot'] and default['confusion_matrix']):                                                                                        
            import seaborn as sns
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix/np.sum(matrix),display_labels=default['display_labels'])
            disp.plot()
            print(colored("----------------------------------------------------------",'red'))

        if(default['classifier']):
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_curve
            from sklearn.metrics import PrecisionRecallDisplay
            #if(default['y_prob']):
            prec, recall, _ = precision_recall_curve(default['y_actual'], default['y_predict'], pos_label=default['classifier'].classes_[1])
            pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
            fpr, tpr, _ = roc_curve(default['y_actual'], default['y_predict'], pos_label=default['classifier'].classes_[1])
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
            
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
            roc_display.plot(ax=ax1)
            pr_display.plot(ax=ax2)
            ax1.title.set_text('ROC Curve')
            ax2.title.set_text('Precision-Recall Curve')
            fig.tight_layout(pad=2.0)
            plt.show()
            print(colored("roc_auc_score: ",'blue'), roc_auc_score(default['y_actual'],default['y_predict'],average=default['average'],
                sample_weight=default['sample_weight'], max_fpr=default['max_fpr'], multi_class=default['multi_class'], labels=default['labels']))
    return process




def descriptive_statistics(func):
    """{describe = True, info = True, statistics = True, null_count = True, memory_usage = True, value_counts= True,
            corelation = True, datasparsity_plot = True, value_co='all', percentiles=None, include=None, exclude=None, 
            datetime_is_numeric=False,verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, 
            null_counts=None,normalize=False, sort=True, ascending=False, bins=None, dropna=True,
            method=pearson, min_periods=1, index=True, deep=False,
            mean_axis=None, mean_skipna=None, mean_level=None, mean_numeric_only=None,
            median_axis=None, median_skipna=None, median_level=None, median_numeric_only=None,
            mode_axis=0, mode_numeric_only=False, mode_dropna=True}"""
    def process(**kwargs):
        default = {'describe' : True, 'info' : True, 'statistics' : True,
        'null_count' : True, 'memory_usage' : True, 'value_counts': True,
        'corelation' : True, 'datasparsity_plot' : True,'value_co' : 'all',
        'percentiles':None, 'include':None, 'exclude':None, 'datetime_is_numeric':False,
       'verbose':None, 'buf':None, 'max_cols':None, 'memory_usage':None, 'show_counts':None, 'null_counts':None,
       'normalize':False, 'sort':True, 'ascending':False, 'bins':None, 'dropna':True,
       'method':'pearson', 'min_periods':1,
        'index':True, 'deep':False,
       'mean_axis':None, 'mean_skipna':None, 'mean_level':None, 'mean_numeric_only':None,
       'median_axis':None, 'median_skipna':None, 'median_level':None, 'median_numeric_only':None,
       'mode_axis':0, 'mode_numeric_only':False, 'mode_dropna':True}
        default.update(kwargs)
        df=func()
        import pandas as pd
        import numpy as np
        from IPython.display import display
        import seaborn as sns
        import matplotlib.pyplot as plt
        if default['describe']:
            print(colored("Describe:\n",'green'))
            
            display(df.describe(percentiles=default['percentiles'], include=default['include'], 
                                exclude=default['exclude'], datetime_is_numeric=default['datetime_is_numeric']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['info']):
            print(colored("Info:\n",'green'))
            display(df.info(verbose=default['verbose'], buf=default['buf'], max_cols=default['max_cols'], 
                            memory_usage=default['memory_usage'], show_counts=default['show_counts'], 
                            null_counts=default['null_counts']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['memory_usage']):
            print(colored("Memory Usage:\n",'green'))
            display(df.memory_usage(index=default['index'], deep=default['index']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['null_count']):
            print(colored("Null Counts:\n",'green'))
            display(df.isnull().sum())
            print(colored("----------------------------------------------------------",'red'))
        if(default['value_counts']):
            x=df.select_dtypes(include='category')
            if(default['value_co']=='all' and len(x.columns)>0):
                print(colored("Value counts of all categorical columns:\n",'green'))
                for col in x:
                    display(df[col].value_counts(normalize=default['normalize'], sort=default['sort'], 
                        ascending=default['ascending'], bins=default['bins'], dropna=default['dropna']))
            elif(isinstance(default['value_co'],list)):
                print(colored("Value counts of specified columns:\n",'green'))
                for col in kwargs['value_co']:
                    display(df[col].value_counts(normalize=default['normalize'], sort=default['sort'], 
                        ascending=default['ascending'], bins=default['bins'], dropna=default['dropna']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['corelation']):
            print(colored("Corelation: \n",'green'))
            correlat = df.corr()
            sns.heatmap(correlat, annot=True)
            plt.title('Correlation Matrix')
            plt.show()
            print(colored("----------------------------------------------------------",'red'))
        if(default['statistics']):
            print(colored("STATISTICS:\n",'green'))
            print(colored("Mean: \n",'blue'))
            display(df.mean(axis=default['mean_axis'], skipna=default['mean_skipna'], 
                level=default['mean_level'], numeric_only=default['mean_numeric_only']))
            print(colored("----------------------------------------------------------",'red'))
            print(colored("Median: \n",'blue'))
            display(df.median(axis=default['median_axis'], skipna=default['median_skipna'], 
                level=default['median_level'], numeric_only=default['median_numeric_only'])) 
            print(colored("----------------------------------------------------------",'red'))
            print(colored("Mode: \n",'blue'))
            display(df.mode(axis=default['mode_axis'], numeric_only=default['mode_numeric_only'], dropna=default['mode_dropna']))
            print(colored("----------------------------------------------------------",'red'))
        if(default['datasparsity_plot']):
            import pandas as pd
            import missingno as msno
            msno.matrix(df)
            msno.bar(df)
            msno.heatmap(df)
    return process