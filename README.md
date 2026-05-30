# agnetic-network-scraper

A network-layer extractor: opens a URL in a browser, captures all network traffic, then runs a PydanticAI investigation agent (backed by `pydantic-graph`) that finds the right JSON endpoint and produces a declarative **JMESPath** or **jq** expression that maps it to your target Pydantic model.
