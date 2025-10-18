import json
import boto3
import numpy as np

def query_endpoint(input_img):
    endpoint_name = 'jumpstart-dft-mx-semseg-fcn-resnet1-20251005-213055'
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-image', Body=input_img, Accept='application/json;verbose')
    return response

def parse_response(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['predictions'],response_dict['labels'], response_dict['image_labels']


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
    with open(img_jpg, 'rb') as file: input_img = file.read()

    try:
        query_response = query_endpoint(input_img)
    except Exception as e:
        if e.response['Error']['Code'] == 'ModelError':
            raise Exception(
                "Backend scripts have been updated in Feb '22 to standardize response "
                "format of endpoint response."
                "Previous endpoints may not support verbose response type used in this notebook."
                f"To use this notebook, please launch the endpoint again. Error: {e}."
            )
        else:
            raise
    try:
        predictions, labels, image_labels =  parse_response(query_response)
    except (TypeError, KeyError) as e:
        raise Exception(
            "Backend scripts have been updated in Feb '22 to standardize response "
            "format of endpoint response."
            "Response from previous endpoints not consistent with this notebook."
            f"To use this notebook, please launch the endpoint again. Error: {e}."
    )