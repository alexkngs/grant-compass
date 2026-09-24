import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grantcompass import professors  # noqa: E402
from grantcompass.professors import (  # noqa: E402
    RateLimited,
    enrich_author,
    fetch_venue_papers,
    last_author_of,
    paper_matches_venue,
)

# Trimmed from real https://api2.openreview.net/notes/search responses (Sept 2026).
SEARCH_RESPONSE = {
    "count": 3,
    "notes": [
        {   # dblp-imported CHI paper: v2 {"value": [...]} shape, authorids mostly dblp URLs
            "id": "ip1ZM9pW6X",
            "content": {
                "venue": {"value": "CHI Extended Abstracts 2017"},
                "venueid": {"value": "dblp.org/conf/CHI/2017"},
                "title": {"value": "CHI 2017 Stories Overview"},
                "authors": {"value": ["Scott P. Robertson", "Geraldine Fitzpatrick", "Doug Zytko"]},
                "authorids": {"value": [
                    "https://dblp.org/search/pid/api?q=author:Scott_P._Robertson:",
                    "https://dblp.org/search/pid/api?q=author:Geraldine_Fitzpatrick:",
                    "https://dblp.org/search/pid/api?q=author:Doug_Zytko:",
                ]},
            },
        },
        {   # dblp record with no author names, only profile ids
            "id": "noNames",
            "content": {
                "venue": {"value": "DIS 2023"},
                "title": {"value": "Designing Things"},
                "authorids": {"value": ["~Some_One1", "~Jane_Q_Doe2"]},
            },
        },
        {   # an official review: no authors at all
            "id": "review1",
            "content": {
                "summary": {"value": "Dichotomous Image Segmentation (DIS) with diffusion"},
                "rating": {"value": 6},
            },
        },
    ],
}


def _response(status, payload):
    resp = mock.Mock(status_code=status)
    resp.json.return_value = payload
    resp.raise_for_status.side_effect = None
    return resp


class TestLastAuthorOf(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(professors.requests, "get",
                               return_value=_response(200, SEARCH_RESPONSE)) as get:
            self.papers = fetch_venue_papers("CHI design", 50)
        self.url = get.call_args.args[0]
        self.params = get.call_args.kwargs["params"]

    def test_uses_search_endpoint(self):
        self.assertEqual(self.url, "https://api2.openreview.net/notes/search")
        self.assertEqual(self.params["term"], "CHI design")

    def test_dblp_v2_shape(self):
        self.assertEqual(last_author_of(self.papers[0]), "Doug Zytko")

    def test_falls_back_to_profile_id(self):
        self.assertEqual(last_author_of(self.papers[1]), "Jane Q Doe")

    def test_review_without_authors(self):
        self.assertIsNone(last_author_of(self.papers[2]))

    def test_v1_plain_list_shape(self):
        self.assertEqual(last_author_of({"content": {"authors": ["A", "B"]}}), "B")

    def test_missing_content(self):
        self.assertIsNone(last_author_of({"content": None}))
        self.assertIsNone(last_author_of({}))

    def test_venue_filter(self):
        self.assertTrue(paper_matches_venue(self.papers[0], "CHI"))
        self.assertTrue(paper_matches_venue(self.papers[1], "DIS"))
        self.assertFalse(paper_matches_venue(self.papers[2], "DIS"))
        self.assertFalse(paper_matches_venue({"content": {"venue": {"value": "ACS Infect. Dis. 2020"}}}, "DIS"))


class TestOpenAlex(unittest.TestCase):
    def test_429_raises(self):
        with mock.patch.object(professors.requests, "get",
                               return_value=_response(429, {"message": "Insufficient budget"})):
            with self.assertRaises(RateLimited):
                enrich_author("Doug Zytko")

    def test_api_key_from_env(self):
        with mock.patch.dict("os.environ", {"OPENALEX_API_KEY": "k"}), \
                mock.patch.object(professors.requests, "get",
                                  return_value=_response(200, {"results": []})) as get:
            enrich_author("Doug Zytko")
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer k"})


if __name__ == "__main__":
    unittest.main()
