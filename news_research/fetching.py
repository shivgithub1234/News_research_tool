"""Bounded public-web fetches with DNS-pinned connections and TLS verification."""
import ipaddress
import socket
from urllib.parse import urlsplit, urljoin, quote

import certifi
import requests
import urllib3

MAX_BYTES = 1_000_000


def public_target(url):
    if len(url) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in url) or "\\" in url:
        raise ValueError("Malformed URL.")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP/HTTPS article URLs without credentials are accepted.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port != (443 if parsed.scheme == "https" else 80):
        raise ValueError("Only standard web ports are accepted.")
    host = parsed.hostname.encode("idna").decode("ascii")
    try:
        addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    except OSError as exc:
        raise requests.RequestException("DNS lookup failed.") from exc
    if not addresses:
        raise ValueError("No public destination found.")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast:
            raise ValueError("Private, local, reserved, and multicast destinations are blocked.")
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        path += "?" + quote(parsed.query, safe="%=&/:?@!$'()*+,;~-._")
    return parsed.scheme, host, addresses[0], port, path


def fetch_html(url):
    for _ in range(4):
        scheme, host, address, port, path = public_target(url)
        kwargs = {"host": address, "port": port, "timeout": urllib3.Timeout(connect=5, read=10), "maxsize": 1}
        if scheme == "https":
            pool = urllib3.HTTPSConnectionPool(**kwargs, assert_hostname=host, server_hostname=host,
                                             cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
        else:
            pool = urllib3.HTTPConnectionPool(**kwargs)
        response = None
        try:
            response = pool.request("GET", path, headers={"Host": host, "User-Agent": "NewsResearchTool/1.0",
                                    "Accept": "text/html,application/xhtml+xml", "Accept-Encoding": "identity"},
                                    redirect=False, retries=False, preload_content=False)
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect has no destination.")
                url = urljoin(url, location)
                continue  # Resolve, validate, and pin every redirect independently.
            if response.status >= 400:
                failed = requests.Response()
                failed.status_code = response.status
                raise requests.HTTPError(response=failed)
            if response.status != 200:
                raise ValueError("Unexpected web response.")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError("The URL did not return an HTML article.")
            if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise ValueError("Compressed responses are not accepted by the bounded fetcher.")
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_BYTES:
                raise ValueError("Article response exceeds the size limit.")
            body = bytearray()
            while True:
                chunk = response.read(min(65536, MAX_BYTES + 1 - len(body)), decode_content=False)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ValueError("Article response exceeds the size limit.")
            # BeautifulSoup handles HTML encoding declarations when given bytes.
            return bytes(body)
        except urllib3.exceptions.HTTPError as exc:
            raise requests.RequestException("Article network request failed.") from exc
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError("Too many redirects.")
