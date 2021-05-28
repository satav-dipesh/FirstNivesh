import streamlit as st

def app():
    import plotly.express as px
    import plotly.graph_objects as go
    from textblob import TextBlob
    import tweepy
    import sys
    import pandas as pd

    api_key = 'q7QHHHAKEwd5igoUvVrx5sCiw'
    api_secret_key = 'i7uhcFirM38bnbYscv32beJnMpsmMxFdYSHitwfSCPIeMj7Lcs'
    access_token = '916414257993879552-kWKlelyL9e6HGH40wcdawT8CiCvO3Hz'
    access_token_secret= 'zYflOPrxysrdOsQiAhp8gmJjAtwRMUcSyX6KlexMk03eB'
    
    auth_handler = tweepy.OAuthHandler(consumer_key = api_key,consumer_secret=api_secret_key)
    auth_handler.set_access_token(access_token,access_token_secret)
    
    api  = tweepy.API(auth_handler)
    
    searchwhat = st.sidebar.text_input("Search Term", 'Zomato IPO')
    tweet_amount = int(st.sidebar.text_input('Tweet Amount', '50'))
    
    tweets = tweepy.Cursor(api.search,q=searchwhat,lang='en').items(tweet_amount)
    
    polarity,positive,negative,neutral = 0,0,0,0
    
    tweetlist = []
    polaritylist = []
    for tweet in tweets:
        final_text = tweet.text.replace('RT','')
        if final_text.startswith(' @'):
            position = final_text.index(':')
            final_text = final_text[position+2:]
        if final_text.startswith('@'):
            position = final_text.index('')
            final_text = final_text[position+2:]
        analysis = TextBlob(final_text)
        tweet_polarity = analysis.polarity
        
        if tweet_polarity>0:
            positive+=1
            polaritylist.append('positive')
        elif tweet_polarity<0:
            negative+=1
            polaritylist.append('negative')
        else:
            neutral+=1
            polaritylist.append('neutral')
        polarity += tweet_polarity
        tweetlist.append(final_text)
        
    labels = ['Positive','Negative','Neutral']
    values = [positive,negative,neutral]
    
    st.write(f'The Sentiment Analysis for Search Term : {searchwhat}')
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    st.plotly_chart(fig)
    
    # tweetcontainer = pd.DataFrame(list(zip(tweetlist,polaritylist)),columns=['Tweets','Sentiment'])
    # st.write(tweetcontainer)

    if len(tweetlist)<10:
        def showTweets(index=0,limit=len(tweetlist)):
            while(index<limit):
                st.write(tweetlist[index])
                st.write(polaritylist[index])
                index+=1
                
        with st.beta_expander('See the Tweets'):
            showTweets()
    
    else:
        def showTweets(index=0,limit=10):
            while(index<limit):
                st.write(tweetlist[index])
                st.write(polaritylist[index])
                index+=1
                
        with st.beta_expander('See the Tweets'):
            showTweets()

    st.subheader('RHP Analysis')
    #st.write('Enter any text from the RHP here')
    user_input = st.text_area('Enter any section from the RHP here')
    rhpanalysis = TextBlob(user_input)
    rhppolarity = rhpanalysis.polarity
    if rhppolarity>0:
        ipo_outlook = 'Positve'
    elif rhppolarity<0:
        ipo_outlook = 'Negative'
    else:
        ipo_outlook = 'Neutral'
    
    with st.beta_expander('See the Analysis'):
        st.subheader(f'Outlook of the Particular IPO is {ipo_outlook} with polarity :{rhppolarity}')        
