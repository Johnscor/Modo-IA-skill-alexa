import os
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Sua chave da OpenRouter (fornecida)
OPENROUTER_API_KEY = "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"

# Configura o cliente OpenAI para usar a OpenRouter
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Modelo a ser usado (DeepSeek via OpenRouter)
MODEL = "deepseek/deepseek-chat"

# Histórico da conversa (inicia com system prompt)
messages = [
    {
        "role": "system",
        "content": "Você é uma assistente muito útil. Por favor, responda de forma clara e concisa em Português do Brasil. Toda resposta deve ter, no máximo, 400 caracteres e o texto deve ser corrido.",
    }
]

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = "Modo I.A Ativado!, Qual a sua pergunta?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

class GptQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        response = generate_gpt_response(query)

        return (
            handler_input.response_builder.speak(response)
            .ask("Você pode fazer uma nova pergunta ou falar: sair.")
            .response
        )

def generate_gpt_response(query):
    try:
        messages.append({"role": "user", "content": query})
        
        # Faz a chamada para a OpenRouter
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
            max_tokens=500,
            # Headers extras recomendados pela OpenRouter (opcionais, mas bons para identificação)
            extra_headers={
                "HTTP-Referer": "https://amazon.com/alexa-skill",
                "X-Title": "IA Alexa Skill"
            }
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"Erro na chamada OpenRouter: {e}", exc_info=True)
        return f"Erro ao gerar resposta: {str(e)}"

class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = "Como posso te ajudar?"
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        speak_output = "Até logo!"
        return handler_input.response_builder.speak(speak_output).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # Qualquer lógica de limpeza aqui
        return handler_input.response_builder.response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Desculpe, não consegui processar sua solicitação."
        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()