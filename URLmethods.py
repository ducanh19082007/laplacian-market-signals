from typing import Dict, List, Optional, Any

#TO BE HONEST, I USED CLAUDE FOR THIS

#only existed for 6 specific Brokers: CoinBase, Gemini, Binance, Kraken, OKX, and Bitstamp
class URL_methods:

    # -------------------------------------------------------------------------
    # Binance
    # -------------------------------------------------------------------------

    @staticmethod
    def make_binance_depth_url(pairs: List[str], depth: int = 5, interval: str = "100ms") -> str:
        return (
            "wss://stream.binance.com:9443/stream?streams="
            + "/".join(f"{p}@depth{depth}@{interval}" for p in pairs)
        )

    # -------------------------------------------------------------------------
    # Coinbase Advanced Trade
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_coinbase_pair(pair: str, assets: List[str]) -> str:
        """Convert 'ethbtc' → 'ETH-BTC' using the known assets list to split."""
        pair = pair.upper()
        assets = [a.upper() for a in assets]
        for quote in assets:
            if pair.endswith(quote):
                base = pair[: -len(quote)]
                if base:
                    return f"{base}-{quote}"
        return pair

    @staticmethod
    def make_coinbase_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Coinbase Advanced Trade WebSocket subscription for the level2 channel.

        Key fix vs the old Coinbase Exchange API:
          - Uses "channel" (singular) instead of "channels" (list).
          - Targets wss://advanced-trade-ws.coinbase.com, not the legacy Pro endpoint.
        """
        formatted_pairs = [URL_methods.binance_to_coinbase_pair(p, assets) for p in pairs]
        return {
            "type": "subscribe",
            "product_ids": [p for p in formatted_pairs if "-" in p],
            "channel": "level2",   # ← singular; "channels" was the old Exchange API
        }

    @staticmethod
    def make_coinbase_payload_extractor():
        """
        Return a *stateful* closure that maintains the full order book across messages.

        Why stateful?
            Coinbase sends a one-time snapshot then only incremental diffs.
            A stateless extractor would discard the snapshot on the first update,
            leaving the book empty.  The closure accumulates price-level changes
            and always returns the current full book to IngestionPipeline.

        Coinbase Advanced Trade l2_data message shape:
        {
          "channel": "l2_data",
          "events": [{
            "type": "snapshot" | "update",
            "product_id": "BTC-USD",
            "updates": [
              {"side": "bid"|"offer", "price_level": "...", "new_quantity": "..."},
              ...
            ]
          }]
        }

        Note: Coinbase uses "offer" (not "sell" or "ask") for the ask side.
        """
        # books[key] = {"bids": {price_str: qty_str}, "asks": {price_str: qty_str}}
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Only handle l2_data; silently ignore subscriptions/heartbeats
            if data.get("channel") != "l2_data":
                return None
            events = data.get("events", [])
            if not events:
                return None

            # A single l2_data message can carry events for several products.
            # Process *all* of them so no incremental diff is dropped, then return
            # the last-updated product's book (the pipeline stores one book per call).
            last_key = None
            for event in events:
                event_type = event.get("type")          # "snapshot" or "update"
                product_id = event.get("product_id", "")
                if not product_id:
                    continue

                key = product_id.replace("-", "").lower()   # "BTC-USD" → "btcusd"

                if event_type == "snapshot":
                    # Full replacement — clear and repopulate
                    books[key] = {"bids": {}, "asks": {}}
                    for upd in event.get("updates", []):
                        price = upd.get("price_level")
                        qty   = upd.get("new_quantity")
                        side  = upd.get("side")
                        if price is None or qty is None:
                            continue
                        if side == "bid":
                            books[key]["bids"][price] = qty
                        elif side == "offer":          # ← Coinbase says "offer", not "ask"
                            books[key]["asks"][price] = qty

                elif event_type == "update":
                    if key not in books:
                        books[key] = {"bids": {}, "asks": {}}
                    for upd in event.get("updates", []):
                        price = upd.get("price_level")
                        qty   = upd.get("new_quantity")
                        side  = upd.get("side")
                        if price is None or qty is None:
                            continue
                        try:
                            qty_f = float(qty)
                        except (ValueError, TypeError):
                            continue
                        if side == "bid":
                            if qty_f == 0:
                                books[key]["bids"].pop(price, None)
                            else:
                                books[key]["bids"][price] = qty
                        elif side == "offer":
                            if qty_f == 0:
                                books[key]["asks"].pop(price, None)
                            else:
                                books[key]["asks"][price] = qty

                if key in books:
                    last_key = key

            if last_key is not None:
                book = books.get(last_key)
                if book:
                    # Return as list-of-tuples; _normalize_quote handles (price, qty) tuples
                    return last_key, {
                        "bids": list(book["bids"].items()),
                        "asks": list(book["asks"].items()),
                    }
            return None

        return extractor

    # -------------------------------------------------------------------------
    # Kraken  (v2 WebSocket API — replaces the old IBKR mock)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_kraken_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ETH/BTC'.

        Kraken uses a slash-separated format with uppercase asset codes.
        Note: Kraken does not list BNB, so BNB pairs will be accepted by this
        converter but will be rejected/ignored by the Kraken server.
        """
        pair_upper = pair.upper()
        assets_upper = [a.upper() for a in assets]
        for quote in assets_upper:
            if pair_upper.endswith(quote):
                base = pair_upper[: -len(quote)]
                if base:
                    return f"{base}/{quote}"
        return pair_upper

    @staticmethod
    def make_kraken_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Kraken v2 WebSocket subscription message for the book channel.

        Kraken v2 shape:
        {
          "method": "subscribe",
          "params": {"channel": "book", "symbol": ["ETH/BTC", ...], "depth": 10}
        }

        After subscribing, Kraken sends one snapshot per symbol, then incremental updates.
        """
        formatted = [URL_methods.binance_to_kraken_pair(p, assets) for p in pairs]
        return {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [p for p in formatted if "/" in p],
                "depth": 10,
            },
        }

    @staticmethod
    def make_kraken_payload_extractor():
        """
        Return a *stateful* closure for the Kraken v2 book channel.

        Kraken v2 book message shape:
        {
          "channel": "book",
          "type": "snapshot" | "update",
          "data": [{
            "symbol": "ETH/BTC",
            "bids": [{"price": 0.0603, "qty": 12.5}, ...],
            "asks": [{"price": 0.0604, "qty": 8.0},  ...],
            "checksum": 1234567890
          }]
        }

        qty == 0 means remove that price level (same convention as Coinbase).
        """
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore heartbeats, subscription acks, etc.
            if data.get("channel") != "book":
                return None
            msg_type = data.get("type")
            if msg_type not in ("snapshot", "update"):
                return None

            data_list = data.get("data", [])
            if not data_list:
                return None

            entry  = data_list[0]
            symbol = entry.get("symbol", "")
            if not symbol:
                return None

            key = symbol.replace("/", "").lower()   # "ETH/BTC" → "ethbtc"

            if msg_type == "snapshot":
                books[key] = {"bids": {}, "asks": {}}
                for b in entry.get("bids", []):
                    books[key]["bids"][str(b["price"])] = str(b["qty"])
                for a in entry.get("asks", []):
                    books[key]["asks"][str(a["price"])] = str(a["qty"])

            elif msg_type == "update":
                if key not in books:
                    books[key] = {"bids": {}, "asks": {}}
                for b in entry.get("bids", []):
                    price = str(b["price"])
                    qty   = b["qty"]
                    if qty == 0:
                        books[key]["bids"].pop(price, None)
                    else:
                        books[key]["bids"][price] = str(qty)
                for a in entry.get("asks", []):
                    price = str(a["price"])
                    qty   = a["qty"]
                    if qty == 0:
                        books[key]["asks"].pop(price, None)
                    else:
                        books[key]["asks"][price] = str(qty)

            book = books.get(key)
            if book:
                return key, {
                    "bids": list(book["bids"].items()),
                    "asks": list(book["asks"].items()),
                }
            return None

        return extractor

    # -------------------------------------------------------------------------
    # OKX  (v5 public WebSocket — wss://ws.okx.com:8443/ws/v5/public)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_okx_pair(pair: str, assets: List[str]) -> str:
        """Convert 'ethbtc' → 'ETH-BTC'. OKX uses dash-separated, uppercase codes."""
        pair = pair.upper()
        assets = [a.upper() for a in assets]
        for quote in assets:
            if pair.endswith(quote):
                base = pair[: -len(quote)]
                if base:
                    return f"{base}-{quote}"
        return pair

    @staticmethod
    def make_okx_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build an OKX v5 subscription for the books5 channel.

        OKX v5 shape:
        {"op": "subscribe",
         "args": [{"channel": "books5", "instId": "BTC-USDT"}, ...]}

        books5 pushes the full top-5 of the book on every update (no incremental
        diffs), so the extractor can be stateless.
        """
        formatted = [URL_methods.binance_to_okx_pair(p, assets) for p in pairs]
        return {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": p} for p in formatted if "-" in p],
        }

    @staticmethod
    def make_okx_payload_extractor():
        """
        Stateless extractor for the OKX v5 books5 channel.

        Unlike Coinbase/Kraken (snapshot + diffs), books5 sends a complete top-5
        snapshot in every message, so there is no book state to accumulate.

        Message shape:
        {
          "arg":  {"channel": "books5", "instId": "BTC-USDT"},
          "data": [{"asks": [["price","size","0","numOrders"], ...],
                    "bids": [["price","size","0","numOrders"], ...],
                    "ts": "...", "seqId": ...}]
        }

        Each level is a list whose first two entries are price and size, which
        _normalize_quote reads directly as (price, qty).
        """
        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore subscribe acks, errors, and pong frames (no "arg"/"data").
            arg = data.get("arg", {})
            if arg.get("channel") != "books5":
                return None
            inst = arg.get("instId", "")
            rows = data.get("data", [])
            if not inst or not rows:
                return None

            entry = rows[0]
            key = inst.replace("-", "").lower()   # "BTC-USDT" → "btcusdt"
            return key, {
                "bids": [(lvl[0], lvl[1]) for lvl in entry.get("bids", []) if len(lvl) >= 2],
                "asks": [(lvl[0], lvl[1]) for lvl in entry.get("asks", []) if len(lvl) >= 2],
            }

        return extractor

    # -------------------------------------------------------------------------
    # Gemini  (v2 marketdata WebSocket — wss://api.gemini.com/v2/marketdata)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_gemini_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ETHBTC'.

        Gemini uses uppercase, concatenated symbols (no separator), which is just
        the Binance lowercase symbol upper-cased. The `assets` arg is unused but
        kept for a uniform converter signature across venues.
        """
        return pair.upper()

    @staticmethod
    def make_gemini_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Gemini v2 marketdata subscription for the level-2 (l2) channel.

        Gemini v2 shape (one frame carries every symbol):
        {"type": "subscribe",
         "subscriptions": [{"name": "l2", "symbols": ["BTCUSD", "ETHBTC", ...]}]}

        IMPORTANT: Gemini errors out (and can drop the connection) on an unknown
        symbol, so callers must pass only symbols Gemini actually lists -- unlike
        Kraken/OKX/Bitstamp which silently ignore unknown channels. The __main__
        block filters to a curated `gemini_pairs` set for exactly this reason.
        """
        symbols = [URL_methods.binance_to_gemini_pair(p, assets) for p in pairs]
        return {
            "type": "subscribe",
            "subscriptions": [{"name": "l2", "symbols": symbols}],
        }

    @staticmethod
    def make_gemini_payload_extractor():
        """
        Return a *stateful* closure for the Gemini v2 l2 channel.

        Gemini l2_updates message shape:
        {
          "type": "l2_updates",
          "symbol": "BTCUSD",
          "changes": [["buy"|"sell", "<price>", "<quantity>"], ...]
        }

        The first l2_updates per symbol is the full snapshot (every level in
        `changes`); subsequent ones are incremental. There is no explicit
        snapshot/update flag, so we just accumulate: a "0" quantity removes the
        level, anything else sets it. (On reconnect Gemini resends a fresh
        snapshot which merges on top of the retained book.)
        """
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            if data.get("type") != "l2_updates":
                return None
            symbol = data.get("symbol", "")
            if not symbol:
                return None

            key = symbol.lower()                       # "BTCUSD" → "btcusd"
            book = books.setdefault(key, {"bids": {}, "asks": {}})

            for change in data.get("changes", []):
                if len(change) < 3:
                    continue
                side, price, qty = change[0], change[1], change[2]
                bucket = (
                    book["bids"] if side == "buy"
                    else book["asks"] if side == "sell"
                    else None
                )
                if bucket is None:
                    continue
                try:
                    qty_f = float(qty)
                except (ValueError, TypeError):
                    continue
                if qty_f == 0:
                    bucket.pop(price, None)
                else:
                    bucket[price] = qty

            return key, {
                "bids": list(book["bids"].items()),
                "asks": list(book["asks"].items()),
            }

        return extractor

    # -------------------------------------------------------------------------
    # Bitstamp  (WebSocket v2 — wss://ws.bitstamp.net, full order_book channel)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_bitstamp_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ethbtc'.

        Bitstamp uses lowercase, concatenated symbols, which is exactly the
        Binance symbol format -- so this is an identity map kept for signature
        uniformity with the other converters.
        """
        return pair.lower()

    @staticmethod
    def make_bitstamp_subscription_messages(pairs: List[str], assets: List[str]) -> list:
        """
        Build Bitstamp subscription frames for the full `order_book` channel.

        Bitstamp wants ONE subscribe frame per channel (it has no batch form),
        so this returns a *list* of messages -- IngestionPipeline sends each in
        turn. Subscribing to a non-existent market yields a harmless bts:error
        event; the connection stays up (unlike Gemini).

        Per-frame shape:
        {"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}}

        The `order_book` channel pushes the full top-100 book on every event, so
        the extractor can stay stateless.
        """
        return [
            {
                "event": "bts:subscribe",
                "data": {"channel": f"order_book_{URL_methods.binance_to_bitstamp_pair(p, assets)}"},
            }
            for p in pairs
        ]

    @staticmethod
    def make_bitstamp_payload_extractor():
        """
        Stateless extractor for the Bitstamp full `order_book` channel.

        Bitstamp data message shape:
        {
          "event": "data",
          "channel": "order_book_btcusd",
          "data": {"timestamp": "...", "microtimestamp": "...",
                   "bids": [["<price>", "<amount>"], ...],
                   "asks": [["<price>", "<amount>"], ...]}
        }

        Every message carries the complete top-100 book, so there is no state to
        accumulate (same idea as OKX books5).
        """
        prefix = "order_book_"

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore subscription acks, errors, heartbeats, reconnect requests.
            if data.get("event") != "data":
                return None
            channel = data.get("channel", "")
            if not channel.startswith(prefix):
                return None

            key = channel[len(prefix):]                # "order_book_btcusd" → "btcusd"
            book = data.get("data", {})
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids and not asks:
                return None

            return key, {
                "bids": [(lvl[0], lvl[1]) for lvl in bids if len(lvl) >= 2],
                "asks": [(lvl[0], lvl[1]) for lvl in asks if len(lvl) >= 2],
            }

        return extractor
