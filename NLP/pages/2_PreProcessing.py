import streamlit as st

st.header('PreProcessing')
st.write('---')
st.write('''The original text from the movie reviews still contains HTML markup, and hasn't been turned
         into "machine friendly" text yet through tokenization, so we need to process the reviews to be
         able to use them. We'll use a simple regex function to remove the HTML, process the text
         into tokens using the porter stemmer algorithmn, and then remove stop words (common words that don't
         provide much meaningful information such as like, and, is ... etc.)''')

text = '''in 1974 the teenager martha moxley maggie grace moves to the high class area of belle haven greenwich connecticut on the mischief night eve of halloween she was murdered in the backyard of her house and her murder remained unsolved twenty two years later the writer mark fuhrman christopher meloni who is a former la detective that has fallen in disgrace for perjury in o j simpson trial and moved to idaho decides to investigate the case with his partner stephen weeks andrew mitchell with the purpose of writing a book the locals squirm and do not welcome them but with the support of the retired detective steve carroll robert forster that was in charge of the investigation in the 70 s they discover the criminal and a net of power and money to cover 

the murder murder in greenwich is a good tv movie with the true story of a murder of a fifteen years old girl that was committed by a wealthy teenager whose mother was a kennedy the powerful and rich family used their influence to cover the murder for more than twenty years however a snoopy detective and convicted perjurer in disgrace was able to disclose how the hideous crime was committed the screenplay shows the investigation of mark and the last days of martha in parallel but there is a lack of the emotion in the dramatization my vote is seven 

title brazil not available
'''

st.text_area('Cleaned Review Exampl',text,height=300)
st.write('''Notice how all of the characters were converted to lowercase, all punctuation was removed,
         and all of the html markdown was removed.''')

stemmed = '''['in',
 '1974',
 'the',
 'teenag',
 'martha',
 'moxley',
 'maggi',
 'grace',
 'move',
 'to',
 'the',
 'high',
 'class',
 'area',
 'of',
 'bell',
 'haven',
 'greenwich',
 'connecticut',
 'on',
 'the',
 'mischief',
 'night',
 'eve',
 'of',
 'halloween',
 'she',
 'wa',
 'murder',
 'in',
 'the',
 'backyard',
 'of',
 'her',
 'hous',
 'and',
 'her',
 'murder',
 'remain',
 'unsolv',
 'twenti',
 'two',
 'year',
 'later',
 'the',
 'writer',
 'mark',
 'fuhrman',
 'christoph',
 'meloni',
 'who',
 'is',
 'a',
 'former',
 'la',
 'detect',
 'that',
 'ha',
 'fallen',
 'in',
 'disgrac',
 'for',
 'perjuri',
 'in',
 'o',
 'j',
 'simpson',
 'trial',
 'and',
 'move',
 'to',
 'idaho',
 'decid',
 'to',
 'investig',
 'the',
 'case',
 'with',
 'hi',
 'partner',
 'stephen',
 'week',
 'andrew',
 'mitchel',
 'with',
 'the',
 'purpos',
 'of',
 'write',
 'a',
 'book',
 'the',
 'local',
 'squirm',
 'and',
 'do',
 'not',
 'welcom',
 'them',
 'but',
 'with',
 'the',
 'support',
 'of',
 'the',
 'retir',
 'detect',
 'steve',
 'carrol',
 'robert',
 'forster',
 'that',
 'wa',
 'in',
 'charg',
 'of',
 'the',
 'investig',
 'in',
 'the',
 '70',
 's',
 'they',
 'discov',
 'the',
 'crimin',
 'and',
 'a',
 'net',
 'of',
 'power',
 'and',
 'money',
 'to',
 'cover',
 'the',
 'murder',
 'murder',
 'in',
 'greenwich',
 'is',
 'a',
 'good',
 'tv',
 'movi',
 'with',
 'the',
 'true',
 'stori',
 'of',
 'a',
 'murder',
 'of',
 'a',
 'fifteen',
 'year',
 'old',
 'girl',
 'that',
 'wa',
 'commit',
 'by',
 'a',
 'wealthi',
 'teenag',
 'whose',
 'mother',
 'wa',
 'a',
 'kennedi',
 'the',
 'power',
 'and',
 'rich',
 'famili',
 'use',
 'their',
 'influenc',
 'to',
 'cover',
 'the',
 'murder',
 'for',
 'more',
 'than',
 'twenti',
 'year',
 'howev',
 'a',
 'snoopi',
 'detect',
 'and',
 'convict',
 'perjur',
 'in',
 'disgrac',
 'wa',
 'abl',
 'to',
 'disclos',
 'how',
 'the',
 'hideou',
 'crime',
 'wa',
 'commit',
 'the',
 'screenplay',
 'show',
 'the',
 'investig',
 'of',
 'mark',
 'and',
 'the',
 'last',
 'day',
 'of',
 'martha',
 'in',
 'parallel',
 'but',
 'there',
 'is',
 'a',
 'lack',
 'of',
 'the',
 'emot',
 'in',
 'the',
 'dramat',
 'my',
 'vote',
 'is',
 'seven',
 'titl',
 'brazil',
 'not',
 'avail']'''

st.text_area('Porter Stemmed Text',stemmed, height=300)
st.write('''Here you can see that the stemmed text becomes a list (or vector) of word stems created according to the Porter algorithm.
         Notice here that not all of the stems identified are even real words. Now we have to remove the stopwords.''')

stemmed_nostop = '''['1974',
 'teenag',
 'martha',
 'moxley',
 'maggi',
 'grace',
 'move',
 'high',
 'class',
 'area',
 'bell',
 'greenwich',
 'connecticut',
 'mischief',
 'night',
 'eve',
 'halloween',
 'wa',
 'murder',
 'backyard',
 'hous',
 'murder',
 'remain',
 'unsolv',
 'twenti',
 'two',
 'year',
 'later',
 'writer',
 'mark',
 'fuhrman',
 'christoph',
 'meloni',
 'former',
 'la',
 'detect',
 'ha',
 'fallen',
 'disgrac',
 'perjuri',
 'j',
 'simpson',
 'trial',
 'move',
 'idaho',
 'decid',
 'investig',
 'case',
 'hi',
 'partner',
 'stephen',
 'week',
 'andrew',
 'mitchel',
 'purpos',
 'write',
 'book',
 'local',
 'squirm',
 'welcom',
 'support',
 'retir',
 'detect',
 'steve',
 'carrol',
 'robert',
 'forster',
 'wa',
 'charg',
 'investig',
 '70',
 'discov',
 'crimin',
 'net',
 'power',
 'money',
 'cover',
 'murder',
 'murder',
 'greenwich',
 'good',
 'tv',
 'movi',
 'true',
 'stori',
 'murder',
 'fifteen',
 'year',
 'old',
 'girl',
 'wa',
 'commit',
 'wealthi',
 'teenag',
 'whose',
 'mother',
 'wa',
 'kennedi',
 'power',
 'rich',
 'famili',
 'use',
 'influenc',
 'cover',
 'murder',
 'twenti',
 'year',
 'howev',
 'snoopi',
 'detect',
 'convict',
 'perjur',
 'disgrac',
 'wa',
 'abl',
 'disclos',
 'hideou',
 'crime',
 'wa',
 'commit',
 'screenplay',
 'show',
 'investig',
 'mark',
 'last',
 'day',
 'martha',
 'parallel',
 'lack',
 'emot',
 'dramat',
 'vote',
 'seven',
 'titl',
 'brazil',
 'avail']'''

st.text_area('Porter Stemmed Text Without Stopwords', stemmed_nostop, height = 300)
st.write('''Now we're ready to do some Natural Language Processing!''')