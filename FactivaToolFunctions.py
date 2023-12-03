import pandas as pd
import matplotlib.pyplot as plt
import locale
import calendar


def make_date_filter(df: pd.DataFrame, start_date: str, end_date: str = None, column: str = "PD" ) -> filter : 
    """
    Create a date filter.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.
    - column (str): Name of the column containing dates (default: 'PD').

    Returns:
    - filter: Filter condition.
    To use this filter on your DataFrame df : 
    filtered_df = df[filter]
    """
    if end_date is not None and start_date is not None :
        filter = (df[column] >= start_date) & (df[column] <= end_date)
    elif end_date is None :
        filter = df[column] >= start_date
    else : 
        filter = df[column] <= end_date
    return filter

def make_words_filter(  df: pd.DataFrame, words_to_match: list[str] = [], words_to_ban: list[str] = [],
                        column: str = "LPTD", match_separator: str = '|', ban_separator: str = '|', case: bool = False) -> filter : 
    """
    Create a word filter.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    - words_to_match (list[str]) : List of words
    - column (str): Name of the column containing the words you want to filter (default: 'LPTD').
    - match_separator (str) : If you want the article having all the words ('&') or at least one of them ('|').
                        Default is '|'. 
    - ban_separator (str) : Same but for discarding words.
    - case (bool) : If you want the case to be respected (default : False).

    Returns:
    - filter: Filter condition.
    To use this filter on your DataFrame df : 
    filtered_df = df[filter]
    """
    filtered_df = df.copy()
    match_filter_df = pd.Series(True, index=filtered_df.index)
    ban_filter_df = pd.Series(True, index=filtered_df.index)

    if words_to_match :
        match_pattern = match_separator.join(words_to_match)
        match_filter_df = filtered_df[column].str.contains(match_pattern, case=case)
    if words_to_ban :
        ban_pattern = ban_separator.join(words_to_ban)
        ban_filter_df = ~filtered_df[column].str.contains(ban_pattern, case=case)

    return match_filter_df & ban_filter_df

def count_article(  df: pd.DataFrame, 
                    start_date: str = None, words_to_match: list[str] = [], words_to_ban: list[str] = [],
                    end_date: str = None, date_column : str = 'PD', words_column:str = 'LPTD', 
                    match_separator: str = '|',  ban_separator: str = '|', 
                    case: bool = False, time_unity: list[str] = ['Year', 'Month'],
                    plot: str = True, topic = 'AI Safety'
                    ) -> list :    
   
    """
    Count the number of article containg some words during a period. Plot it by month (default).

    Parameters:
    - df (DataFrame): Input DataFrame.
    - start_date/end_date (str): Start/end date in the format 'YYYY-MM-DD'.
    - date_column (str): Name of the column containing dates (default: 'PD').
    - words_column (str): Name of the column containing the words you want to filter (default: 'LPTD').
    - separator (str) : If you want the article having all the words ('&') or at least one of them ('|').
                        Default is ('|').
    - case (bool) : If you want the case to be respected (default : False).
    - time_unity (list[str]): Don't change it for the moment please.
    - plot (bool) : If True, it plots the figure. (default : True)

    Returns:
    - The DataFrame filtered by the time period and words you chose if you chose some.
    """
    #We copy to avoid changing the original DataFrame.
    new_df = df.copy()

    if start_date is not None or end_date is not None :
        period_filter = make_date_filter(new_df, start_date, end_date, date_column)
        new_df = new_df[period_filter]
    
    if words_to_match or words_to_ban :
        words_filter = make_words_filter(new_df, words_to_match, words_to_ban, words_column, match_separator, ban_separator, case)
        new_df = new_df[words_filter]
    
    final_df = new_df

    for unit in time_unity :
        if unit == 'Day':
            return "Day not available for the moment."
            final_df[unit] = final_df[date_column].dt.day
        if unit == 'Month':
            final_df[unit] = final_df[date_column].dt.month
        if unit == 'Year':
            final_df[unit] = final_df[date_column].dt.year
        
    # Group by year and month, count the number of items for each month
    items_by_time_unity = final_df.groupby(time_unity).size().reset_index(name='Count')

    if plot :
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

        items_by_time_unity['Year-Month'] = items_by_time_unity['Year'].astype(str) + '-' + items_by_time_unity['Month'].astype(str)

        # Plotting the number of items by month
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(items_by_time_unity.index, items_by_time_unity['Count'], color='blue', marker='o')

        # Adding month names above each point on the plot
        for i, label in enumerate(items_by_time_unity['Year-Month']):
            year, month = label.split('-')
            month_name = calendar.month_abbr[int(month)]
            plt.text(i, items_by_time_unity['Count'][i] + 0.5, month_name, ha='center', va='bottom')


        # Set x-axis ticks to display years
        ticks_indices = items_by_time_unity[items_by_time_unity['Month'] % 4 == 0].index
        plt.xticks(ticks=ticks_indices, labels=items_by_time_unity.loc[ticks_indices, 'Year'])
        
        containing_str = ''
        within = ''
        if words_to_match : 
            joined_words = ', '.join(words_to_match)
            containing_str = f'containing the word(s) : "{joined_words}", '
        if start_date and end_date:
            within = f'within the period {start_date} - {end_date}, '
        elif start_date :
            within = f'within the period {start_date} - 2023-11-01, '
        elif end_date :
            within = f'within the period 2018-11-01 - {end_date}, '
        plt.title(f'Number of {topic} articles {containing_str}{within}monthly')
        plt.xlabel('Years')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    final_df.pop('Year')
    final_df.pop('Month')
    return final_df

def time_group_articles(  df: pd.DataFrame, 
                    start_date: str = None, 
                    words_to_match: list[str] = [], words_to_ban: list[str] = [], 
                    end_date: str = None, date_column : str = 'PD', words_column:str = 'LPTD', 
                    match_separator: str = '|', ban_separator: str = '|',
                    case: bool = False, time_unity: str = 'Year',
                    ) -> list :    
   
    """
    Count the number of article containg some words during a period. Plot it by month (default).

    Parameters:
    - df (DataFrame): Input DataFrame.
    - start_date/end_date (str): Start/end date in the format 'YYYY-MM-DD'.
    - date_column (str): Name of the column containing dates (default: 'PD').
    - words_column (str): Name of the column containing the words you want to filter (default: 'LPTD').
    - separator (str) : If you want the article having all the words ('&') or at least one of them ('|').
                        Default is ('|').
    - case (bool) : If you want the case to be respected (default : False).
    - time_unity (str): 'Year', 'Month' (default = 'Year'). Whether you want it to be grouped yearly or monthly.

    Returns:
    - The DataFrame grouped by the time period and filtered by the words you chose if you chose some.
    """
    #We copy to avoid changing the original DataFrame.
    new_df = df.copy()

    if start_date is not None or end_date is not None :
        period_filter = make_date_filter(new_df, start_date, end_date, date_column)
        new_df = new_df[period_filter]
    
    if words_to_match or words_to_ban :
        words_filter = make_words_filter(new_df, words_to_match, words_to_ban, words_column, match_separator, ban_separator, case)
        new_df = new_df[words_filter]
    
    if time_unity == 'Year' :
        new_df[time_unity] = new_df['PD'].dt.year.astype(str)
    elif time_unity == 'Month' :
        new_df[time_unity] = new_df['PD'].dt.month.astype(str)
    
    # Group by year and month, count the number of items for each month
    grouped_texts = new_df.groupby(time_unity)['LPTD'].apply(lambda x: ' '.join(x)).reset_index()

    return grouped_texts


import altair as alt
# import altair_viewer
import numpy as np

def plot_TF_IDF(top_tfidf, words_in_column: bool = True, red_dotted_terms: list[str] = []):
    # Terms in this list will get a red dot in the visualization
    term_list = red_dotted_terms
    # adding a little randomness to break ties in term ranking
    top_tfidf_plusRand = top_tfidf.copy()
    top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_tfidf.shape[0])*0.0001
    # base for all visualizations, with rank calculation
    x_encoding = 'document:N' if words_in_column else 'rank:O'
    y_encoding = 'rank:O' if words_in_column else 'document:N'

    base = alt.Chart(top_tfidf_plusRand).encode(
        x=x_encoding,
        y=y_encoding
    ).transform_window(
        rank="rank()",
        sort=[alt.SortField("tfidf", order="descending")],
        groupby=["document"],
    )
    
    # heatmap specification
    heatmap = base.mark_rect().encode(
        color = 'tfidf:Q'
    )
    # red circle over terms in above list
    circle = base.mark_circle(size=100).encode(
        color = alt.condition(
            alt.FieldOneOfPredicate(field='term', oneOf=term_list),
            alt.value('red'),
            alt.value('#FFFFFF00')        
        )
    )
    # text labels, white for darker heatmap colors
    text = base.mark_text(baseline='middle').encode(
        text = 'term:N',
        color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
    )
    # display the three superimposed visualizations
    chart = (heatmap + circle + text).properties(width = 600)
    #altair_viewer.show(chart) #This does not work yet, I have a trouble with installed version of altair_viewer apparently.
    return chart

from sklearn.feature_extraction.text import TfidfVectorizer

def stopwords_from_path(stopwords_path, split = ',') :
    stop_word_text = open(stopwords_path)
    content = stop_word_text.read()
    return content.split(split)
    

def TF_IDF_Factiva( df: pd.DataFrame, 
                    index_column: str = 'Year', text_column: str = 'LPTD',
                    words_in_column: bool = True, stop_words: str = 'english',
                    top_k: int = 15,
                    plot: bool = True, red_dotted_terms: list[str] = []) :
    """
    Please make sure all your columns have string items (if it is years for instance).

    stop_words : 
        - 'english' (default): standard set of skilearn english stopwords
        - a path to another file of stopwords.
    """
    if stop_words != 'english': stop_words = stopwords_from_path(stop_words)
    
    tfidf_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words= stop_words, smooth_idf=True, norm='l2')

    tfidf_vector = tfidf_vectorizer.fit_transform(df[text_column])
   

    if plot : 
        tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df[index_column], columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_df = tfidf_df.stack().reset_index()

        tfidf_df = tfidf_df.rename(columns={0:'tfidf', index_column: 'document','level_1': 'term', 'level_2': 'term'})
        top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(top_k)
        chart = plot_TF_IDF(top_tfidf, words_in_column, red_dotted_terms)
        return chart
    else : 
        tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=df[index_column], columns=tfidf_vectorizer.get_feature_names_out())
        #tfidf_df = tfidf_df.stack().reset_index()

        # Renaming the columns properly
        tfidf_df = tfidf_df.rename(columns={0: 'tfidf', 'level_0': index_column, 'level_1': 'term'})

        # Sorting the TF-IDF values within each document (year)
        top_tfidf = tfidf_df.sort_values(by=[index_column, 'tfidf'], ascending=[True, False]).groupby(index_column).head(top_k)
        return top_tfidf
    

def plot_topics(components, terms, top_k=10, red_dotted_terms=[]):
    term_list = red_dotted_terms
    
    # Get the top terms for each topic from the LDA model
    topics = []
    for index, topic in enumerate(components):
        topic_terms = [terms[i] for i in topic.argsort()[:-top_k - 1:-1]]
        topics.append(topic_terms)

    # Create a DataFrame for visualization
    top_topics_df = pd.DataFrame({
        'topic': np.repeat(np.arange(len(topics)), top_k),
        'rank': np.tile(np.arange(top_k), len(topics)),
        'term': np.concatenate(topics)
    })

    # Base for all visualizations
    base = alt.Chart(top_topics_df).encode(
        x=alt.X('topic:O'),
        y=alt.Y('rank:O')
    )

    # Red circle over terms in the above list
    circle = base.mark_circle(size=100).encode(
        color=alt.condition(
            alt.FieldOneOfPredicate(field='term', oneOf=term_list),
            alt.value('red'),
            alt.value('#FFFFFF00')
        )
    )

    # Text labels
    text = base.mark_text(baseline='middle').encode(
        text='term:N',
        color=alt.condition(
            alt.FieldOneOfPredicate(field='term', oneOf=term_list),
            alt.value('black'),
            alt.value('black')
        )
    )

    # Display the superimposed visualizations
    chart = (circle + text).properties(width=600)
    return chart


from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import RegexpTokenizer

def LDA_Topic_Modeling( df: pd.DataFrame, 
                    text_column: str = 'LPTD', stop_words: str = 'english',
                    top_k: int = 15, n_topics = 5,
                    plot: bool = True, red_dotted_terms: list[str] = []):
    """
    Latent Dirichlet Allocation Topic Modeling

    Please make sure all your columns have string items (if it is years for instance).

    stop_words can either be : 
        - 'english' (default): standard set of skilearn english stopwords
        - A path to another file of stopwords.
    """
    if stop_words != 'english': stop_words = stopwords_from_path(stop_words)

    tokenizer = RegexpTokenizer(r'\w+')

    tfidf = TfidfVectorizer(input='content', tokenizer=tokenizer.tokenize, analyzer='word', stop_words= stop_words, ngram_range=(1,1), smooth_idf=True, norm='l2')

    train_data = tfidf.fit_transform(df[text_column])  #not sure about the to_list()

    # Create LDA object
    model=LatentDirichletAllocation(n_components=n_topics)

    # Fit and Transform SVD model on data
    model.fit_transform(train_data)

    # Get Components 
    lda_components=model.components_
    terms = tfidf.get_feature_names_out()
    if plot : 
        return plot_topics(lda_components, terms, top_k=top_k, red_dotted_terms=red_dotted_terms)
    else : 
        return lda_components, terms
