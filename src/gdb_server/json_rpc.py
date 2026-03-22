"""JSON-RPC 2.0 encoder/decoder."""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .codes import PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR


def encode_response(
    request_id: Optional[Union[int, str]],
    result: Any,
) -> str:
    """
    Encode a JSON-RPC 2.0 response (success).

    Args:
        request_id: The request ID to respond to
        result: The result data

    Returns:
        JSON-RPC 2.0 response string
    """
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
    return json.dumps(response)


def encode_error_response(
    request_id: Optional[Union[int, str]],
    code: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Encode a JSON-RPC 2.0 error response.

    Args:
        request_id: The request ID (can be None for parse errors)
        code: Error code (negative integer)
        message: Human-readable error message
        data: Optional additional error data

    Returns:
        JSON-RPC 2.0 error response string
    """
    error = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data

    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }
    return json.dumps(response)


def encode_notification(method: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Encode a JSON-RPC 2.0 notification (request without id).

    Args:
        method: The method name
        params: Optional parameters

    Returns:
        JSON-RPC 2.0 notification string
    """
    notification = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params
    return json.dumps(notification)


def encode_batch_response(responses: List[str]) -> str:
    """Encode a batch response."""
    # Parse each response, collect into array
    parsed = []
    for resp_str in responses:
        resp = json.loads(resp_str)
        if resp.get("result") is not None or resp.get("error") is not None:
            parsed.append(resp)
    return json.dumps(parsed) if parsed else "[]"


def decode_request(request_str: str) -> Dict[str, Any]:
    """
    Decode a JSON-RPC 2.0 request.

    Args:
        request_str: The JSON-RPC request string

    Returns:
        Parsed request dict

    Raises:
        ValueError: If the request is invalid
    """
    try:
        request = json.loads(request_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")

    # Validate JSON-RPC version
    if request.get("jsonrpc") != "2.0":
        raise ValueError("Invalid JSON-RPC version")

    # Must have method
    if "method" not in request:
        raise ValueError("Missing 'method' field")

    # Check for invalid fields
    valid_fields = {"jsonrpc", "method", "params", "id"}
    for field in request:
        if field not in valid_fields:
            raise ValueError(f"Invalid field: {field}")

    return request


def decode_notification(request_str: str) -> Dict[str, Any]:
    """
    Decode a JSON-RPC 2.0 notification (request without id).

    Args:
        request_str: The JSON-RPC notification string

    Returns:
        Parsed notification dict

    Raises:
        ValueError: If the notification is invalid
    """
    request = decode_request(request_str)

    # Notifications must not have an id
    if "id" in request:
        raise ValueError("Notification must not have an 'id' field")

    return request


def decode_batch_request(request_str: str) -> List[Dict[str, Any]]:
    """
    Decode a batch JSON-RPC 2.0 request.

    Args:
        request_str: The JSON-RPC batch request string

    Returns:
        List of parsed request dicts

    Raises:
        ValueError: If the batch is invalid
    """
    try:
        requests = json.loads(request_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")

    if not isinstance(requests, list):
        raise ValueError("Batch request must be an array")

    if not requests:
        raise ValueError("Empty batch request")

    # Validate each request
    validated = []
    for i, req in enumerate(requests):
        if not isinstance(req, dict):
            raise ValueError(f"Batch request[{i}] must be an object")

        if req.get("jsonrpc") != "2.0":
            raise ValueError(f"Batch request[{i}] has invalid JSON-RPC version")

        if "method" not in req:
            raise ValueError(f"Batch request[{i}] missing 'method' field")

        validated.append(req)

    return validated


def create_error_response(
    request_id: Optional[Union[int, str]],
    code: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 error response dict (without encoding).

    Args:
        request_id: The request ID
        code: Error code
        message: Error message
        data: Optional error data

    Returns:
        Error response dict ready for json.dumps()
    """
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def create_success_response(
    request_id: Optional[Union[int, str]],
    result: Any,
) -> Dict[str, Any]:
    """
    Create a JSON-RPC 2.0 success response dict (without encoding).

    Args:
        request_id: The request ID
        result: The result data

    Returns:
        Success response dict ready for json.dumps()
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }