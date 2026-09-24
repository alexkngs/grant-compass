import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grantcompass import professors  # noqa: E402
from grantcompass.professors import (  # noqa: E402
    RateLimited,
    enrich_author,
    enrich_author_by_id,
    fetch_openalex_venue_works,
    fetch_venue_papers,
    last_author_of,
    last_authorship_of,
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


# Trimmed from a real https://api.openalex.org/works response (Sept 2026).
OPENALEX_WORKS_RESPONSE = {
    "results": [
        {
            "title": "Co-designing with older adults",
            "authorships": [
                {"author": {"id": "https://openalex.org/A1", "display_name": "First Author"}},
                {"author": {"id": "https://openalex.org/A5072280320", "display_name": "Panayiotis Koutsabasis"}},
            ],
        },
        {"title": "No authors", "authorships": []},
    ],
}


class TestOpenAlexVenues(unittest.TestCase):
    def test_query_filters_source_and_years(self):
        with mock.patch.object(professors.requests, "get",
                               return_value=_response(200, OPENALEX_WORKS_RESPONSE)) as get:
            works = fetch_openalex_venue_works("S152445846", "design", [2025, 2026], 500)
        self.assertEqual(get.call_args.args[0], "https://api.openalex.org/works")
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["filter"], "primary_location.source.id:S152445846,publication_year:2025|2026")
        self.assertEqual(params["search"], "design")
        self.assertEqual(params["per_page"], 200)
        self.assertEqual(len(works), 2)

    def test_accepts_full_source_url(self):
        with mock.patch.object(professors.requests, "get",
                               return_value=_response(200, {"results": []})) as get:
            fetch_openalex_venue_works("https://openalex.org/S152445846", "", [], 50)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["filter"], "primary_location.source.id:S152445846")
        self.assertNotIn("search", params)

    def test_last_authorship(self):
        works = OPENALEX_WORKS_RESPONSE["results"]
        self.assertEqual(last_authorship_of(works[0]),
                         {"name": "Panayiotis Koutsabasis", "id": "https://openalex.org/A5072280320"})
        self.assertIsNone(last_authorship_of(works[1]))

    def test_enrich_by_id_uses_author_endpoint(self):
        author = {"display_name": "Panayiotis Koutsabasis", "id": "https://openalex.org/A5072280320",
                  "last_known_institutions": [{"display_name": "University of the Aegean", "country_code": "GR"}],
                  "summary_stats": {"h_index": 20}, "works_count": 100}
        with mock.patch.object(professors.requests, "get", return_value=_response(200, author)) as get:
            record = enrich_author_by_id("https://openalex.org/A5072280320")
        self.assertEqual(get.call_args.args[0], "https://api.openalex.org/authors/A5072280320")
        self.assertEqual(record["country_code"], "GR")
        self.assertEqual(record["h_index"], 20)

    def test_run_uses_openalex_venues(self):
        cfg = {"search": {"openalex_venues": [{"name": "Design Studies", "id": "S152445846"}]},
               "applicant": {"field_keywords": ["design"]}}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(professors, "DATA_DIR", Path(tmp)), \
                mock.patch.object(professors, "load_config", return_value=cfg), \
                mock.patch.object(professors, "fetch_openalex_venue_works",
                                  return_value=OPENALEX_WORKS_RESPONSE["results"]), \
                mock.patch.object(professors, "enrich_author") as by_name, \
                mock.patch.object(professors, "enrich_author_by_id",
                                  return_value={"name": "P. K.", "country_code": "GR"}) as by_id:
            records = professors.run()
        by_id.assert_called_once_with("https://openalex.org/A5072280320")
        by_name.assert_not_called()
        self.assertEqual(records[0]["source_venue"], "Design Studies")
        self.assertEqual(records[0]["source_paper_title"], "Co-designing with older adults")


class TestRunWarnings(unittest.TestCase):
    """The key can be injected by a proxy, so run() must not warn just because the env var is unset."""

    def _run(self, enrich):
        cfg = {"search": {"venues": ["CHI"]}, "applicant": {"field_keywords": ["design"]}}
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(professors, "_warned", set()), \
                mock.patch.object(professors, "DATA_DIR", Path(tmp)), \
                mock.patch.object(professors, "load_config", return_value=cfg), \
                mock.patch.object(professors, "fetch_venue_papers", return_value=SEARCH_RESPONSE["notes"]), \
                mock.patch.object(professors, "enrich_author", side_effect=enrich), \
                mock.patch("sys.stderr", stderr):
            records = professors.run()
        return records, stderr.getvalue()

    def test_no_warning_when_openalex_answers(self):
        records, err = self._run(lambda name: {"name": name, "country_code": "AT"})
        self.assertEqual(len(records), 1)
        self.assertEqual(err, "")

    def test_warns_with_key_hint_on_429(self):
        records, err = self._run(RateLimited("HTTP 429"))
        self.assertEqual(records, [])
        self.assertIn("rate limit", err)
        self.assertIn("OPENALEX_API_KEY", err)


if __name__ == "__main__":
    unittest.main()
