from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape
import pylatex as pl

import pandas as pd
import matplotlib.pyplot as plt  # noqa
from pylatex import Document, Section, Figure, NoEscape
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.








class PDF:
    def __init__(self,path):
        self.path = path
        self.df = pd.read_csv(f'uploads/{self.path}',header=0,sep=';',parse_dates=['Time',"Time.1"]).sort_values("Time")

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

        min_down_by_symbol_pd = df.groupby(df.Symbol)['Profit'].min(numeric_only=True).sort_values(ascending=False)
        min_down_by_symbol = min_down_by_symbol_pd.to_latex()


        

        
        mean_down_by_symbol_pd = df.groupby(df.Symbol)['Profit'].mean(numeric_only=True).sort_values(ascending=False)
        mean_down_by_symbol = mean_down_by_symbol_pd.to_latex()

        

        
        mean_down_by_day_pd =  df.groupby(df.Time.dt.day_name())['Profit'].mean()
        mean_down_by_day =  mean_down_by_day_pd.to_latex()

        
        min_down_by_day_pd =  df.groupby(df.Time.dt.day_name())['Profit'].min()
        min_down_by_day =  min_down_by_day_pd.to_latex()


        mean_down_by_hour_pd =  df.groupby(df.Time.dt.hour)['Profit'].mean()
        mean_down_by_hour = mean_down_by_hour_pd.to_latex()


        
        min_down_by_hour_pd =  df.groupby(df.Time.dt.hour)['Profit'].min()
        min_down_by_hour =  min_down_by_hour_pd.to_latex()

        tmp_df = df.sort_values("Time").copy()
        tmp_df.reset_index(drop=True,inplace=True)
        tmp_df = tmp_df.fillna(0)
        tmp_df['ret'] = tmp_df['Profit']+tmp_df['Commission'] + tmp_df['Swap']
        tmp_df['amount'] = abs(tmp_df['Profit'] /(tmp_df['Price']-tmp_df['Price.1']))
        # tmp_df.loc[0,'ret'] = 0

        tmp_df['balance'] = tmp_df['ret'].cumsum()
        tmp_df.loc[0,'ret'] = 0
        tmp_df['return_ptc'] = tmp_df['ret'].cumsum()* 100/tmp_df['balance'][0] 
        self.tmp_df = tmp_df
        self.downturn_df = self.count_downturns(tmp_df)

        balance_df = tmp_df.groupby(tmp_df.Time.dt.date)['balance'].mean()
        ptc_df = tmp_df.groupby(tmp_df.Time.dt.date)['return_ptc'].mean()
        

        with doc.create(Section('Balance overview')):
            with doc.create(Figure(position='htbp')) as plot:
                plt.figure(figsize=(20,8))
                plt.plot(balance_df,linewidth=3,color='#02bfe0')
                plt.ylabel("Dollars")
                plt.xlabel("Date")
                plt.grid(axis='y')
                plot.add_plot()
                plt.close()

            with doc.create(Figure(position='htbp')) as plot:
                plt.figure(figsize=(20,8))
                plt.plot(ptc_df,linewidth=3,color='#02bfe0')
                plt.ylabel("%")
                plt.xlabel("Date")
                plt.grid(axis='y')
                plot.add_plot()
                plt.close()





        with doc.create(Section('By symbol')):

            with doc.create(Subsection('Minimum profit ')):
                doc.append(NoEscape(min_down_by_symbol))
                with doc.create(Figure(position='htbp')) as plot:
                    df = pd.DataFrame(min_down_by_symbol_pd)

                    mask1 = df[df.Profit > 0]
                    mask2 = df[df.Profit < 0]
                    plt.figure(figsize=(20,15))
                    plt.bar(mask1.index,mask1.Profit,color='limegreen')
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

        doc.append(NoEscape('\\newpage '))
        with doc.create(Section("Downturns")):
            with doc.create(Figure(position='htbp')) as plot:
                plt.figure(figsize=(20,15))
                plt.grid(axis='x')
                plt.grid(axis='y')
                
                # self.downturn_df = self.downturn_df.dropna()
                # self.downturn_df.Downturn_pct.plot.bar()
                try:
                    plt.bar(self.downturn_df.Time, self.downturn_df.Downturn_pct)
                    plt.ylabel('%')
                    plt.title("Max Downturns")
                    plot.add_plot()
                    plt.close()
                except:
                    plot.add_plot()
                    plt.title("NO DATA")
                    plt.close()


                
            with doc.create(Figure(position='htbp')) as plot:
                plt.figure(figsize=(20,15))
                plt.grid(axis='x')
                plt.grid(axis='y')

                # self.downturn_df = self.downturn_df.dropna()
                # self.downturn_df.Downturn_pct.plot.bar()
                try:
                    plt.ylabel('Dollars')
                    plt.title("Max Downturns")
                    plt.bar(self.downturn_df.Time, self.downturn_df.Downturn_cash)
                    plot.add_plot()
                    plt.close()
                except:
                    plot.add_plot()
                    plt.title("NO DATA")
                    plt.close()


                

        
    def count_downturns(self,df):
        df = df.copy()
        for symbol in df.Symbol.unique():
            print(symbol)
            try:
                search_df = pd.read_csv(f"{symbol}.csv",parse_dates=['date'], header=0)
            except:
                continue
            df.loc[:,'Downturn_pct'] = 0 
            df.loc[:,'Downturn_cash'] = 0


            crop_df = df.loc[df.Symbol==f'{symbol}']    
            crop_df = crop_df.reset_index(drop=True,)
            crop_df = crop_df[["Type","Time",'Time.1','Price',"Price.1","ret","balance",'amount']]
            crop_df.columns = ['side','start_time','end_time','start_price','end_price','profit','balance','amount']  # type: ignore
            for row in crop_df.itertuples():
                start_time= row.start_time
                end_time = row.end_time
                start_price = row.start_price

                side = row.side
                
                amount = row.amount
                balance = row.balance
                idx = row.Index
                down,dollar = self.find_downturns(search_df,start_time,end_time,start_price,side,amount,balance)
                
                if down:
                    df.loc[idx,'Downturn_pct'] = max(down)
                    df.loc[idx,'Downturn_cash'] = max(dollar)
                else:
                    df.loc[idx,'Downturn_pct'] = 0
                    df.loc[idx,'Downturn_cash'] = 0
            
        return df
        


            

    def find_downturns(self,search_df,start_time,end_time,start_price,side,amount,balance):
        search_set = search_df.loc[(search_df.date>=start_time) & (search_df.date <= end_time)]
        downs = []
        dollars = []

        for row in search_set.itertuples(index=False):

            if side == 'Buy':
                downs.append(
                    round((start_price*amount-row.low*amount)/balance*100,4)
                )
                dollars.append(start_price*amount - row.low*amount)
            elif side == 'Sell':
                downs.append(
                    round((row.high*amount - start_price*amount)/balance*100,4)
                )
                dollars.append(row.high*amount - start_price*amount)
                
        return downs,dollars


                    

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
        return new_path
