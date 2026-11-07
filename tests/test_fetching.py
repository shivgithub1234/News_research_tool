import socket
import unittest
from unittest.mock import Mock, patch
from news_research.fetching import public_target, fetch_html, MAX_BYTES


def address(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class FetchingTests(unittest.TestCase):
    @patch("news_research.fetching.socket.getaddrinfo")
    def test_private_loopback_metadata_and_mixed_answers_rejected(self, dns):
        for ip in ["127.0.0.1", "10.1.1.1", "169.254.169.254", "::1", "192.168.1.2", "224.0.0.1"]:
            dns.return_value = address(ip)
            with self.subTest(ip=ip), self.assertRaises(ValueError):
                public_target("https://example.com/article")
        dns.return_value = address("8.8.8.8") + address("127.0.0.1")
        with self.assertRaises(ValueError):
            public_target("https://example.com/article")

    def test_scheme_credentials_port_and_control_characters_rejected(self):
        for url in ["file:///secret", "http://user:password@example.com/", "https://example.com:8080/", "https://example.com/\r\nHost:evil", "https://example.com\\@localhost/"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                public_target(url)

    @patch("news_research.fetching.urllib3.HTTPSConnectionPool")
    @patch("news_research.fetching.socket.getaddrinfo", return_value=address("8.8.8.8"))
    def test_connection_pins_validated_ip_and_verifies_original_tls_hostname(self, dns, factory):
        response = Mock(status=200, headers={"Content-Type": "text/html"})
        response.read.side_effect = [b"<html>Article</html>", b""]
        factory.return_value.request.return_value = response
        self.assertEqual(fetch_html("https://example.com/story"), b"<html>Article</html>")
        self.assertEqual(factory.call_args.kwargs["host"], "8.8.8.8")
        self.assertEqual(factory.call_args.kwargs["assert_hostname"], "example.com")
        self.assertEqual(factory.call_args.kwargs["cert_reqs"], "CERT_REQUIRED")
        self.assertEqual(dns.call_count, 1)
        response.close.assert_called_once()

    @patch("news_research.fetching.urllib3.HTTPSConnectionPool")
    @patch("news_research.fetching.socket.getaddrinfo", side_effect=[address("8.8.8.8"), address("127.0.0.1")])
    def test_redirect_is_revalidated(self, dns, factory):
        factory.return_value.request.return_value = Mock(status=302, headers={"Location": "https://internal.example/secret"})
        with self.assertRaises(ValueError):
            fetch_html("https://example.com/story")
        self.assertEqual(factory.call_count, 1)

    @patch("news_research.fetching.urllib3.HTTPSConnectionPool")
    @patch("news_research.fetching.socket.getaddrinfo", return_value=address("8.8.8.8"))
    def test_oversized_and_non_html_bodies_rejected(self, dns, factory):
        for headers in [{"Content-Type": "text/html", "Content-Length": str(MAX_BYTES + 1)},
                        {"Content-Type": "application/octet-stream"},
                        {"Content-Type": "text/html", "Content-Encoding": "gzip"}]:
            factory.return_value.request.return_value = Mock(status=200, headers=headers)
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                fetch_html("https://example.com/story")
