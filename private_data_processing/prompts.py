# convert document to event
doc2event_prompt = """Given a document, please extract all the events and their associated topics in the context.

Note: 
1. The event should not contain ambiguous references, such as ’he’,’ she,’ and ’it’, and should use complete names. 
2. You should give at least one passage in the original news associated to the event you extract, DO NOT make up any event.
3. If there are multiple paragraphs associated to the extracted event, please list and number all of them.
4. If the event does not contain some of the arguments mentioned above, please leave it empty.
5. The type of Event involves fine-grained events and general events, where fine-grained events focus on specific facts and details while general events are summarizations of happened fine-grained events.
6. Please return the fine-grained events first, then return general events.
The document is:
{doc}

Please return the extracted event in the following format with following arguments:
[Event]:
[Topic]:
[Original context]: 1.
2.
...
[Type]:

Events you extract are:

"""

# convert event to query and answer
event2qa_prompt = """
Given several events and their original source document, please ask several questions according to the infomation and give the original reference paragraph following this format:
[Envent]:
[Question]:
Note: 1. Don't need to mention all the arguments in your question. 
2. You can involve the original document information, but make sure that your question is about the topic of the given event.
3. You should ask questions separately to different events.
4. Don't return repeat original context.

Document:
{doc}
Event:
{event}
Your question towards given event:


"""


# keyword masking
keyword_extraction_prompt = """
Given a query and a paragraph including the answer of the query, please extract all the common keywords that query and paragraph both have:
Note:
1. The definition of keywords is: words in the query and paragraph that are particularly distinctive and related to the main topic. Less important pronouns or frequently occurring words do not fall into this category.
2. The words you extract must appear in both the query and the paragraph.
3. Do not output other format, just list all the words as follows:
investigation, Eastwood, Filing

Query:
{query}
Paragraph:
{paragraph}
keywords:

"""