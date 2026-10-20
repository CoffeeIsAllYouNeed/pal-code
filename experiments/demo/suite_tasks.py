"""The 30 experiments: thirty different things people build with AI agents. Each is a short multi-step job (a brief, a spec the person
attaches, then follow-up requests), run once as a plain chat client and once through Pal with the same model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM = "You are a careful senior engineer and writer. Do exactly what is asked, keep answers complete but compact."


@dataclass(frozen=True)
class SuiteTask:
    id: int
    category: str
    title: str
    brief: str
    spec: tuple                       # the requirements the person attaches as a reference file
    followups: tuple = field(default_factory=tuple)

    def spec_text(self) -> str:
        return f"Requirements for: {self.title}\n" + "\n".join(f"- {line}" for line in self.spec)

    def steps(self) -> list:
        return [self.brief, *self.followups]


_COMMON = ("Add input validation and clear error messages.", "Add a short usage example.", "Review what you wrote and fix any bug you find.")


def _t(i, cat, title, brief, spec, follow=_COMMON):
    return SuiteTask(i, cat, title, brief, tuple(spec), tuple(follow))


TASKS: tuple = (
    # web pages and apps
    _t(1, "Web", "Hotel landing page", "Build a single-file HTML landing page for a small hotel.",
       ["Hero, rooms, gallery, contact form", "Mobile friendly", "No external libraries", "Calm blue and white colours", "Accessible: alt text, labels"]),
    _t(2, "Web", "To-do web app", "Build a to-do app in one HTML file with plain JavaScript.",
       ["Add, tick, delete tasks", "Keep tasks in localStorage", "Filter all/active/done", "Keyboard friendly", "No libraries"]),
    _t(3, "Web", "Portfolio site", "Build a one-page portfolio site for a designer.",
       ["About, projects grid, contact", "Dark and light theme toggle", "Smooth scrolling", "Semantic HTML", "Under 200 lines of CSS"]),
    _t(4, "Web", "Tic-tac-toe game", "Build tic-tac-toe in one HTML file.",
       ["Two players on one screen", "Detect win and draw", "Restart button", "Highlight the winning line", "No libraries"]),
    _t(5, "Web", "Quiz app", "Build a multiple-choice quiz app in one HTML file.",
       ["10 questions in a JS array", "Score at the end", "One question per screen", "Show correct answer after each", "No libraries"]),
    _t(6, "Web", "Recipe card page", "Build a recipe page with an ingredient scaler.",
       ["Servings input rescales amounts", "Print friendly CSS", "Steps as an ordered list", "Fractions shown nicely", "No libraries"]),
    # command line and scripts
    _t(7, "Scripts", "CSV cleaner", "Write a Python CLI that cleans a CSV file.",
       ["Trim spaces, drop empty rows, remove duplicates", "Normalise column names to snake_case", "Report what changed", "argparse interface", "Standard library only"]),
    _t(8, "Scripts", "Log analyser", "Write a Python script that summarises a web server log.",
       ["Count requests per status code", "Top 10 paths and slowest 10 requests", "Handle malformed lines", "Print a text report", "Standard library only"]),
    _t(9, "Scripts", "Unit converter CLI", "Write a Python command-line unit converter.",
       ["Length, mass, temperature, volume", "Parse input like '5 km to mi'", "Reject unknown units", "Unit tests with unittest", "Standard library only"]),
    _t(10, "Scripts", "File organiser", "Write a Python script that sorts a folder's files into sub-folders by type.",
       ["Dry-run mode by default", "Never overwrite existing files", "Log every move", "Configurable type map", "Standard library only"]),
    _t(11, "Scripts", "Expense tracker", "Write a Python expense tracker that stores data in a JSON file.",
       ["Add, list and delete expenses", "Monthly totals per category", "Export to CSV", "argparse subcommands", "Standard library only"]),
    _t(12, "Scripts", "Markdown to HTML", "Write a small Python Markdown to HTML converter.",
       ["Headings, bold, italic, links, lists, code blocks", "Escape HTML safely", "Read a file, write a file", "Unit tests", "No third-party packages"]),
    # backend and data
    _t(13, "Backend", "REST API sketch", "Design and write a REST API for a library (books and loans) using Flask.",
       ["Endpoints for books, members, loans", "JSON errors with status codes", "In-memory storage", "Simple pagination", "OpenAPI-style summary"]),
    _t(14, "Backend", "SQL schema", "Design a SQL schema for an online shop and write the queries it needs.",
       ["Customers, products, orders, order items", "Primary and foreign keys, indexes", "Query: top sellers, monthly revenue", "PostgreSQL syntax", "Explain each index"]),
    _t(15, "Backend", "API client wrapper", "Write a Python client class for a weather REST API.",
       ["Retry with backoff on 5xx", "Timeout on every call", "Typed return objects", "Cache for 10 minutes", "Use requests"]),
    _t(16, "Backend", "JSON schema validator", "Write a tiny JSON schema validator in Python.",
       ["Types, required, enum, min/max, nested objects", "Return all errors with paths", "No third-party packages", "Unit tests", "Docstrings"]),
    _t(17, "Backend", "Task scheduler", "Write a Python task scheduler that runs jobs at intervals.",
       ["Interval and one-shot jobs", "Thread based", "Graceful stop", "Log failures and keep going", "Standard library only"]),
    _t(18, "Backend", "Rate limiter", "Write a token-bucket rate limiter in Python.",
       ["Per-key buckets", "Thread safe", "Injectable clock for tests", "Unit tests", "Standard library only"]),
    # AI and agent work
    _t(19, "AI agents", "RAG engine", "Write a small retrieval engine in Python for question answering over text files.",
       ["Split documents into chunks", "TF-IDF scoring, no vector database", "Return top-k chunks with sources", "Cite file names in answers", "Standard library only"]),
    _t(20, "AI agents", "Chatbot system prompt", "Design a system prompt and test conversations for a customer-support chatbot for a bank.",
       ["Never reveal account details without verification", "Escalate fraud reports to a human", "Friendly, short answers", "Refuse legal advice", "Include 6 test conversations"]),
    _t(21, "AI agents", "Tool-calling agent loop", "Write a Python agent loop that lets a model call tools.",
       ["Tool registry with JSON schemas", "Max 5 steps", "Validate arguments before calling", "Log every call", "Mock model for tests"]),
    _t(22, "AI agents", "Prompt test harness", "Write a harness that runs a list of prompts against a model function and scores the answers.",
       ["Cases as a list of dicts", "Checks: contains, regex, max length", "Pass rate per case", "Text report", "Standard library only"]),
    _t(23, "AI agents", "Meeting-notes summariser", "Write a Python tool that turns raw meeting notes into decisions and action items.",
       ["Output sections: summary, decisions, actions with owners", "Detect dates and names", "Plain text and Markdown output", "Handle empty input", "Standard library only"]),
    # writing and documents
    _t(24, "Writing", "Product README", "Write a README for an open-source command-line note-taking tool.",
       ["Install, quick start, commands table", "Configuration section", "Contributing guide", "Licence section", "Friendly tone"]),
    _t(25, "Writing", "Outreach emails", "Write a sequence of three outreach emails to sponsors for a community event.",
       ["Short subject lines", "Under 120 words each", "One clear ask per email", "Polite follow-up timing", "No hype words"]),
    _t(26, "Writing", "Changelog generator", "Write a Python script that builds a changelog from commit messages.",
       ["Group by feat, fix, docs", "Keep a Changelog format", "Version headers from tags", "Read from a text file", "Standard library only"]),
    _t(27, "Writing", "Translator helper", "Write a Python helper that prepares text for translation and checks the result.",
       ["Protect placeholders like {name}", "Keep line breaks", "Check placeholders survive", "Report mismatches", "Standard library only"]),
    # engineering practice
    _t(28, "Engineering", "Test writer", "Here is a function: def slugify(text): return '-'.join(text.lower().split()). Write thorough unit tests, then improve the function.",
       ["Unicode, punctuation, empty strings", "pytest style", "Keep the function pure", "Explain each edge case", "No third-party packages"]),
    _t(29, "Engineering", "Code review", "Review this function and rewrite it: def avg(l): return sum(l)/len(l).",
       ["Cover empty lists, non-numbers, large lists", "Type hints and docstring", "Explain every change", "Keep it dependency free", "Add tests"]),
    _t(30, "Engineering", "Regex helper", "Write Python functions that extract emails, phone numbers and dates from text with regular expressions.",
       ["Indian and international phone formats", "ISO and dd/mm/yyyy dates", "No false positives on version numbers", "Unit tests", "Explain each pattern"]),
)

assert len(TASKS) == 30 and [t.id for t in TASKS] == list(range(1, 31))
