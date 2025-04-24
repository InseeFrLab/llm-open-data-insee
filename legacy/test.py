from shiny import render, module, reactive
from shiny import ui, App
import faicons as fa
import asyncio
import json
from chatlas import ChatOpenAI
from loguru import logger

ICONS = {
    "positive": fa.icon_svg("thumbs-up"),
    "negative": fa.icon_svg("thumbs-down"),
    "stop": fa.icon_svg("stop")
}
welcome = "Posez moi des questions"
system_prompt = "répond en français"

chat_client = ChatOpenAI(
    base_url="https://projet-models-hf-vllm.user.lab.sspcloud.fr/v1",
    api_key="EMPTY",
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    system_prompt=system_prompt,
)

def stop_button():
    return ui.input_action_button(
        "stop", "", icon=ICONS.get("stop"), class_="btn-link border-0"
    )


#@module.ui
def gen_button(prefix):
    return ui.TagList([
        ui.input_action_button(
                f"positive", "", icon = ICONS.get("positive"), class_="btn-link border-0"
            ),
        ui.input_action_button(
                f"negative", "", icon = ICONS.get("negative"), class_="btn-link border-0"
            )
    ])


app_ui = ui.page_fillable(
    ui.panel_title("Hello Shiny Chat"),
    ui.chat_ui("chat"),
    # Important : déclaration de l'input stop
    ui.input_action_button("stop", "", style="display:none"),
    ui.output_text("value"),
    fillable_mobile=True,
)

@module.server
def row_server(input, output, session):
    @output
    # @render.ui
    # def buttonsOK():
    #     return ui.TagList([
    #         gen_button("mess1")
    #     ])
    @render.ui
    def text_out():
        return f'You entered "{input.positive()}"'


def server(input, output, session):

    chat = ui.Chat(id="chat", messages=[welcome])
    val = reactive.value(0)

    @reactive.effect
    @reactive.event(input.stop)
    def _():
        chat.latest_message_stream.cancel()
        ui.notification_show("Stream cancelled", type="warning")

    @reactive.effect
    def _():
        ui.update_action_button(
            "cancel",
            disabled=chat.latest_message_stream.status() != "running"
        )

    @render.download(filename="messages.json", label="Download messages")
    def download():
        yield json.dumps(chat.messages())

    # # Define a callback to run when the user submits a message
    # @chat.on_user_submit
    # async def handle_user_input(user_input: str):
    #     # Append a response to the chat
    #     async with chat.message_stream_context() as outer:
    #         await outer.append("Starting stream 🔄...\n\nProgress:")
    #         async with chat.message_stream_context() as inner:
    #             for x in [0, 50, 100]:
    #                 await inner.replace(f" {x}%\n\n{stop_button()}")
    #                 await asyncio.sleep(1)
    #         await outer.append(f"\n\n{gen_button("row")}")

    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        async with chat.message_stream_context() as outer:

            async with chat.message_stream_context() as inner:
                full_text = ""
                stream = await chat_client.stream_async(user_input)

                async for chunk in stream:
                    full_text += chunk
                    await inner.replace(f"{full_text}\n\n{stop_button()}")

                # Une fois le stream terminé, on remplace le message pour retirer le bouton
                await inner.replace(full_text)

            await outer.append(f"\n\n{gen_button('row')}")


# def server(input, output, session):

#     with ui.hold() as df_ui:
#     @render.ui
#     def buttonsOK():
#         return ui.TagList([
#             gen_button("mess1")
#         ])

#     @chat.on_user_submit
#     async def _():
#         async with chat.message_stream_context() as outer:
#             await outer.append("Starting stream 🔄...\n\nProgress:")
#             async with chat.message_stream_context() as inner:
#                 for x in [0, 50, 100]:
#                     await inner.replace(f" {x}%")
#                     await asyncio.sleep(1)
#             await outer.append(df_ui)


app = App(app_ui, server)

