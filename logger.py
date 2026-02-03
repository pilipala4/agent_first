import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_calls.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _log_api_call_start(model: str, messages_count: int):
    """记录 API 调用开始"""
    logger.info(f"开始调用 LLM API: model={model}, messages_count={messages_count}")


def _log_api_call_success(model: str, tokens_used: int):
    """记录 API 调用成功"""
    logger.info(f"LLM API 调用成功: model={model}, tokens_used={tokens_used}")
