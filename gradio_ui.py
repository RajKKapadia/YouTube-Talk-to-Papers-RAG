import gradio as gr

from arxiv_call import download_and_save_papers
from index import create_index
from conversation import handle_user_query, create_conversation


with gr.Blocks() as arxiv_ui:

    with gr.Row(
        variant='compact'
    ):
        search_input = gr.components.Textbox(label='Search query')
        numb_papers_input = gr.components.Number(label='Number of papers', value=5.0)
        result_input = gr.components.Textbox(label='Download status')
    search_btn = gr.components.Button(value='Search')
    search_btn.click(
        download_and_save_papers,
        [search_input, numb_papers_input],
        [result_input]
    ).then(
        create_index,
        [search_input],
        [result_input]
    )

with gr.Blocks() as chat_ui:
    chatbot = gr.Chatbot(label='Talk to the Douments', bubble_full_width=False)
    msg = gr.Textbox(label='Query', placeholder='Enter text and press enter')
    clear = gr.ClearButton([msg, chatbot], variant='stop')

    msg.submit(
        handle_user_query,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        create_conversation,
        [chatbot],
        [chatbot]
    )

demo = gr.TabbedInterface(
    [arxiv_ui, chat_ui],
    ['Search ARXIV Papers', 'Chat with Papers']
)

demo.queue()
