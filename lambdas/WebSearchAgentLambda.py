
import logging
import json
from typing import Dict, Any, Tuple
from http import HTTPStatus
import urllib.parse
import urllib.request
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing Bedrock agent requests.
    
    Args:
        event (Dict[str, Any]): The Lambda event containing action details
        context (Any): The Lambda context object
    
    Returns:
        Dict[str, Any]: Response containing the action execution results
    
    Raises:
        KeyError: If required fields are missing from the event
    """
    try:
        action_group = event['actionGroup']
        function = event['function']
        message_version = event.get('messageVersion',1)
        parameters = event.get('parameters', [])
        search_query, include_images = extract_parameters(parameters)
        tavApiKey = get_tavily_api_key()
        searchResponse = tavily_ai_search(search_query, tavApiKey, include_images=include_images)

        # Execute your business logic here. For more information, 
        # refer to: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html
        response_body = {
            'TEXT': {
                'body': searchResponse
            }
        }
        action_response = {
            'actionGroup': action_group,
            'function': function,
            'functionResponse': {
                'responseBody': response_body
            }
        }
        response = {
            'response': action_response,
            'messageVersion': message_version
        }

        logger.info('Response: %s', response)
        return response

    except KeyError as e:
        logger.error('Missing required field: %s', str(e))
        return {
            'statusCode': HTTPStatus.BAD_REQUEST,
            'body': f'Error: {str(e)}'
        }
    except Exception as e:
        logger.error('Unexpected error: %s', str(e))
        return {
            'statusCode': HTTPStatus.INTERNAL_SERVER_ERROR,
            'body': 'Internal server error'
        }

def get_tavily_api_key() -> str:
    secret_name = "prod/tavily/apiKey"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    return json.loads(get_secret_value_response['SecretString'])['TAVILY_API_KEY']

def extract_parameters(parameters: Any) -> Tuple[str, bool]:
    """
    Extract and validate the SearchQuery and optional ImageSearchEnabled parameters.
    """
    if not isinstance(parameters, list):
        raise KeyError("'parameters' must be provided as a list.")

    params_by_name: Dict[str, Any] = {}
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name = param.get('name')
        if not name:
            continue
        params_by_name[name] = param.get('value')

    if 'SearchQuery' in params_by_name:
        search_query_value = params_by_name['SearchQuery']
    elif parameters and isinstance(parameters[0], dict) and 'value' in parameters[0]:
        search_query_value = parameters[0]['value']
    else:
        raise KeyError("SearchQuery")

    if not isinstance(search_query_value, str):
        search_query_value = str(search_query_value)

    include_images_value = params_by_name.get('ImageSearchEnabled', False)
    include_images = normalize_to_bool(include_images_value)

    return search_query_value, include_images


def normalize_to_bool(value: Any) -> bool:
    """
    Normalize the various ways a boolean can be represented in the incoming payload.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def tavily_ai_search(search_query: str, api_key: str, include_images: bool = False) -> str:
    logger.info(f"executing Tavily AI search with {search_query=}")

    base_url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "api_key": api_key,
        "query": search_query,
        "search_depth": "basic",
        "include_images": include_images,
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 5,
        "include_domains": [],
        "exclude_domains": [],
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url, data=data, headers=headers)  # nosec: B310 fixed url we want to open

    try:
        response = urllib.request.urlopen(request, timeout=30)  # nosec: B310 fixed url we want to open
        response_data: str = response.read().decode("utf-8")
        logger.debug(f"response from Tavily AI search {response_data=}")
        return response_data
    except urllib.error.HTTPError as e:
        logger.error(f"failed to retrieve search results from Tavily AI Search, error: {e.code}")

    return ""
