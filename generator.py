from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape
import pylatex as pl

import pandas as pd
import matplotlib.pyplot as plt  # noqa
from pylatex import Document, Section, Figure, NoEscape
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.








class PDF:
    def __init__(self,path):
        self.path = path
        self.df = pd.read_csv(f'uploads/{self.path}',header=0,sep=';',parse_dates=['Time'])

    def fill_document(self,doc):

        df = self.df.copy()
        try:
            df['Profit'] = pd.to_numeric(df['Profit'].str.replace(' ',''))
        except:
            pass
        plt.rc('axes', titlesize=20)     
        plt.rc('axes', labelsize=20)    
        plt.rc('xtick', labelsize=20)    
        plt.rc('ytick', labelsize=20)    
        plt.rc('legend', fontsize=20)    
        plt.rcParams['axes.titlesize'] = 45

        min_down_by_symbol = df.groupby(df.Symbol)['Profit'].min(numeric_only=True).sort_values(ascending=False).to_latex()
        min_down_by_symbol_pd = df.groupby(df.Symbol)['Profit'].min(numeric_only=True).sort_values(ascending=False)

        

        mean_down_by_symbol = df.groupby(df.Symbol)['Profit'].mean(numeric_only=True).sort_values(ascending=False).to_latex()
        mean_down_by_symbol_pd = df.groupby(df.Symbol)['Profit'].mean(numeric_only=True).sort_values(ascending=False)

        

        mean_down_by_day =  df.groupby(df.Time.dt.day_name())['Profit'].mean().to_latex()
        mean_down_by_day_pd =  df.groupby(df.Time.dt.day_name())['Profit'].mean()

        min_down_by_day =  df.groupby(df.Time.dt.day_name())['Profit'].min().to_latex()
        min_down_by_day_pd =  df.groupby(df.Time.dt.day_name())['Profit'].min()


        mean_down_by_hour =  df.groupby(df.Time.dt.hour)['Profit'].mean().to_latex()
        mean_down_by_hour_pd =  df.groupby(df.Time.dt.hour)['Profit'].mean()

        min_down_by_hour =  df.groupby(df.Time.dt.hour)['Profit'].min().to_latex()
        min_down_by_hour_pd =  df.groupby(df.Time.dt.hour)['Profit'].min()
        
        

    


        with doc.create(Section('By symbol')):

            with doc.create(Subsection('Minimum profit ')):
                doc.append(NoEscape(min_down_by_symbol))
                with doc.create(Figure(position='htbp')) as plot:
                    df = pd.DataFrame(min_down_by_symbol_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    bar = plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')

                    plt.xticks(rotation=90)
                    plt.ylabel('Dollars')
                    plt.title("Min profit")
                    plt.grid(axis='y')


                    plot.add_plot()
                    plt.close()

            doc.append(NoEscape('\\newpage '))
            with doc.create(Subsection('Mean profit ')):
                doc.append(NoEscape(mean_down_by_symbol))
                with doc.create(Figure(position='htbp')) as plot:
                    df = pd.DataFrame(mean_down_by_symbol_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')
                    plt.ylabel('Dollars')
                    plt.title("Mean profit")
                    plt.grid(axis='y')


                    plt.xticks(rotation=90)
                    plot.add_plot()
                    plt.close()
        doc.append(NoEscape('\\newpage '))
        
        with doc.create(Section('By day')):
            with doc.create(Subsection('Minimum profit ')):
                doc.append(NoEscape(min_down_by_day))
                with doc.create(Figure(position='htbp')) as plot:

                    # cat_size_order = pd.CategoricalDtype(
                    #     ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'], 
                    #     ordered=True)

                    # df = pd.DataFrame(min_down_by_day_pd)
                    # x = df.loc[:, ['Profit']]
                    # df['mpg_z'] = (x)/(x.max()-x.min())
                    # df['colors'] = ['lightcoral' if x < 0 else 'limegreen' for x in df['mpg_z']]

                    # df.reset_index(inplace=True)
                    # df['Time'] = df['Time'].astype(cat_size_order)
                    # df.sort_values("Time",inplace=True)
                    # df.reset_index(inplace=True)


                    # # Draw plots
                    # plt.figure(figsize=(14,14), dpi= 80)
                    # plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z)
                    # for x, y, tex in zip(df.mpg_z, df.index, df.Profit):
                    #     t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if tex < 0 else 'left', 
                    #                 verticalalignment='center', fontdict={'color':'lightcoral' if tex < 0 else 'limegreen', 'size':14})

                    # # Decorations    
                    # plt.yticks(df.index, df.Time, fontsize=12)
                    # plt.title('Min profit', fontdict={'size':20})
                    # plt.grid(linestyle='--', alpha=0.5)
                    # plt.xlim(-4, 4)
                    # plt.ylabel('Day')
                    # plt.xlabel('Dollars')
                    # plt.title("Min profit")
                    # plt.grid(axis='y')

                    # plot.add_plot()
    
                    # plt.close()
                    df = pd.DataFrame(min_down_by_day_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')
                    plt.ylabel('Dollars')
                    plt.title("Mean profit")
                    plt.grid(axis='y')


                    plt.xticks(rotation=90)
                    plot.add_plot()
                    plt.close()

            doc.append(NoEscape('\\newpage '))

            with doc.create(Subsection('Mean profit ')):
                doc.append(NoEscape(mean_down_by_day))
                with doc.create(Figure(position='htbp')) as plot:

                    # cat_size_order = pd.CategoricalDtype(
                    #     ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'], 
                    #     ordered=True)

                    # df = pd.DataFrame(mean_down_by_day_pd)
                    # x = df.loc[:, ['Profit']]
                    # df['mpg_z'] = (x)/(x.max()-x.min())
                    # df['colors'] = ['lightcoral' if x < 0 else 'limegreen' for x in df['mpg_z']]

                    # df.reset_index(inplace=True)
                    # df['Time'] = df['Time'].astype(cat_size_order)
                    # df.sort_values("Time",inplace=True)
                    # df.reset_index(inplace=True)


                    # # Draw plots
                    # plt.figure(figsize=(14,14), dpi= 80)
                    # plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z)
                    # for x, y, tex in zip(df.mpg_z, df.index, df.Profit):
                    #     t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if tex < 0 else 'left', 
                    #                 verticalalignment='center', fontdict={'color':'lightcoral' if tex < 0 else 'limegreen', 'size':14})

                    # # Decorations    
                    # plt.yticks(df.index, df.Time, fontsize=12)
                    # plt.title('Mean profit', fontdict={'size':20})
                    # plt.grid(linestyle='--', alpha=0.5)
                    # plt.xlim(-4, 4)
                    # plt.ylabel('Day')
                    # plt.xlabel('Dollars')
                    # plt.title("Mean profit")
                    # plt.grid(axis='y')


                    # plot.add_plot()
    
                    # plt.close()
                    df = pd.DataFrame(mean_down_by_day_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')
                    plt.ylabel('Dollars')
                    plt.title("Mean profit")
                    plt.grid(axis='y')


                    plt.xticks(rotation=90)
                    plot.add_plot()
                    plt.close()
                    


        doc.append(NoEscape('\\newpage '))
        with doc.create(Section('By hour')):
            with doc.create(Subsection('Minimum profit ')):
                doc.append(NoEscape(min_down_by_hour))
                with doc.create(Figure(position='htbp')) as plot:
                    df = pd.DataFrame(min_down_by_hour_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')
                    plt.ylabel('Dollars')
                    plt.title("Min profit")

                    plt.xticks(range(0,24, 1),rotation=90)
                    plt.grid(axis='y')

                    
                    plot.add_plot()
                    plt.close()
            
            doc.append(NoEscape('\\newpage '))
            with doc.create(Subsection('Mean profit ')):
                doc.append(NoEscape(mean_down_by_hour))
                with doc.create(Figure(position='htbp')) as plot:
                    df = pd.DataFrame(mean_down_by_hour_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
                    plt.bar(mask2.index,mask2.Profit,color='lightcoral')
                    plt.ylabel('Dollars')
                    plt.title("Mean profit")

                    plt.xticks(range(0,24, 1),rotation=90)
                    plt.grid(axis='y')

                    plot.add_plot()
                    plt.close()
                    

    def generate_pdf(self):    
        geometry_options = {"right": "2cm", "left": "2cm"}
        doc = Document(geometry_options=geometry_options)

        doc.preamble.append(Command('title', f'Trader id: {self.path.split(".")[0]}'))
        doc.preamble.append(Command('author', 'Test sccript_v.2'))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))

        self.fill_document(doc)
        new_path = f'uploads/{self.path[:-4]}'
        doc.generate_pdf(new_path, clean_tex=True)
        print("-"*100)
        print(new_path)
        return new_path
