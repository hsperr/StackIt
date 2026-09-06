"""The online mode: hosting, joining, seat ownership and catching up.

These drive the JSON API through Flask's test client, which is the same path a
browser takes, so they cover the seat tokens and the move replay a poll gets.
"""
import pytest

import server


@pytest.fixture
def client():
    server.games.clear()
    server.codes.clear()
    return server.app.test_client()


def host_game(client, players=2, size=5, name="Host"):
    j = client.post("/api/game", json={"mode": "online", "players": players,
                                       "size": size, "name": name}).get_json()
    return j["state"], j["token"]


def join(client, code, name):
    r = client.post("/api/join", json={"code": code, "name": name})
    return r, r.get_json()


def test_hosting_hands_out_a_code_and_seat_one(client):
    st, token = host_game(client, players=3)
    assert len(st["code"]) == server.CODE_LEN
    assert st["your_seat"] == 1 and st["is_host"]
    assert st["started"] is False
    assert [p["joined"] for p in st["players"]] == [True, False, False]


def test_the_game_starts_when_the_last_seat_fills(client):
    st, _ = host_game(client, players=3)
    _, a = join(client, st["code"], "Ana")
    assert a["state"]["started"] is False
    _, b = join(client, st["code"], "Bo")
    assert b["state"]["started"] is True
    assert b["state"]["your_seat"] == 3


def test_the_code_is_not_case_sensitive(client):
    st, _ = host_game(client, players=2)
    r, _ = join(client, st["code"].lower(), "Ana")
    assert r.status_code == 200


def test_a_full_game_turns_away_the_next_arrival(client):
    st, _ = host_game(client, players=2)
    join(client, st["code"], "Ana")
    r, j = join(client, st["code"], "Late")
    assert r.status_code == 409 and "already started" in j["error"]


def test_only_the_seat_whose_turn_it_is_may_move(client):
    st, host = host_game(client, players=2)
    _, a = join(client, st["code"], "Ana")
    iid, ana = st["iid"], a["token"]

    r = client.post(f"/api/game/{iid}/move", json={"x": 0, "y": 0, "token": ana})
    assert r.status_code == 409 and "not your turn" in r.get_json()["error"]

    r = client.post(f"/api/game/{iid}/move", json={"x": 0, "y": 0})
    assert r.status_code == 403

    r = client.post(f"/api/game/{iid}/move", json={"x": 0, "y": 0, "token": host})
    assert r.status_code == 200 and r.get_json()["state"]["current_player"] == 2


def test_nobody_may_move_before_the_last_player_arrives(client):
    st, host = host_game(client, players=3)
    r = client.post(f"/api/game/{st['iid']}/move", json={"x": 0, "y": 0, "token": host})
    assert r.status_code == 409 and "waiting" in r.get_json()["error"].lower()


def test_a_poll_replays_the_moves_it_missed(client):
    st, host = host_game(client, players=2)
    _, a = join(client, st["code"], "Ana")
    iid, ana = st["iid"], a["token"]
    client.post(f"/api/game/{iid}/move", json={"x": 0, "y": 0, "token": host})

    j = client.get(f"/api/game/{iid}?since=0&token={ana}").get_json()
    assert [e["n"] for e in j["events"]] == [1]
    assert j["events"][0]["move"] == [0, 0]
    assert j["events"][0]["frames"], "a move must carry the frames to animate"
    assert j["state"]["your_seat"] == 2


def test_a_browser_too_far_behind_gets_no_events_only_the_position(client):
    st, host = host_game(client, players=2)
    _, a = join(client, st["code"], "Ana")
    iid, ana = st["iid"], a["token"]
    tokens = {1: host, 2: ana}
    for i in range(server.MAX_EVENTS + 2):
        state = client.get(f"/api/game/{iid}?token={host}").get_json()["state"]
        mv = state["legal"][0]
        client.post(f"/api/game/{iid}/move",
                    json={"x": mv[0], "y": mv[1], "token": tokens[state["current_player"]]})
    j = client.get(f"/api/game/{iid}?since=0&token={ana}").get_json()
    assert j["events"] == []
    assert j["state"]["move_count"] == server.MAX_EVENTS + 2


def test_the_host_can_start_with_whoever_turned_up(client):
    st, host = host_game(client, players=4)
    _, a = join(client, st["code"], "Ana")
    iid = st["iid"]

    assert client.post(f"/api/game/{iid}/start",
                       json={"token": a["token"]}).status_code == 403
    j = client.post(f"/api/game/{iid}/start", json={"token": host}).get_json()
    assert j["state"]["started"] and j["state"]["num_players"] == 2
    assert len(j["state"]["players"]) == 2


def test_a_host_alone_cannot_start(client):
    st, host = host_game(client, players=3)
    r = client.post(f"/api/game/{st['iid']}/start", json={"token": host})
    assert r.status_code == 400


def test_an_online_game_has_no_take_backs(client):
    st, host = host_game(client, players=2)
    _, a = join(client, st["code"], "Ana")
    iid = st["iid"]
    client.post(f"/api/game/{iid}/move", json={"x": 0, "y": 0, "token": host})
    assert client.get(f"/api/game/{iid}?token={host}").get_json()["state"]["can_undo"] is False
    assert client.post(f"/api/game/{iid}/undo", json={"token": host}).status_code == 403


def test_a_wrong_code_says_so(client):
    r = client.post("/api/join", json={"code": "ZZZZ", "name": "Nobody"})
    assert r.status_code == 404


def test_only_a_humans_only_game_may_be_big_or_crowded(client):
    hvh = client.post("/api/game", json={"mode": "hvh", "players": 5,
                                         "size": server.MAX_BOARD_HUMANS}).get_json()["state"]
    assert hvh["num_players"] == 5
    assert hvh["size_x"] == server.MAX_BOARD_HUMANS

    hva = client.post("/api/game", json={"mode": "hva", "ai": "alphabeta", "players": 5,
                                         "size": server.MAX_BOARD_HUMANS}).get_json()["state"]
    assert hva["num_players"] == 2
    assert hva["size_x"] == server.MAX_BOARD


def test_a_knocked_out_player_is_reported_as_out(client):
    """Player 2 is boxed in with no empty cell left, so the state says so."""
    from board import Board
    st, host = host_game(client, players=3, size=2)
    _, a = join(client, st["code"], "Ana")
    _, b = join(client, st["code"], "Bo")
    g = server.get_game(st["iid"])
    g.board = Board.from_custom_board([[1, 1], [1, 1]], [[1, 3], [3, 3]],
                                      current_player=1, num_players=3)
    g.moves = [(0, 0)]
    players = g.state(host)["players"]
    assert [p["out"] for p in players] == [False, True, False]


def test_lobbies_cannot_push_engine_games_out_of_memory(client):
    """The two kinds of game are capped separately: an engine game carries an
    MCTS tree worth tens of megabytes, a lobby about one."""
    engine = client.post("/api/game", json={"mode": "hva", "ai": "alphabeta"}).get_json()
    iid = engine["state"]["iid"]
    for _ in range(server.MAX_HUMAN_GAMES + 5):
        client.post("/api/game", json={"mode": "online", "players": 2})
    assert server.get_game(iid) is not None, "a lobby flood evicted the engine game"
    lobbies = [g for g in server.games.values() if g.mode == "online"]
    assert len(lobbies) == server.MAX_HUMAN_GAMES


def test_an_untouched_game_is_forgotten_and_frees_its_code(client):
    st, _ = host_game(client, players=2)
    server.get_game(st["iid"]).touched -= server.GAME_TTL + 1
    client.post("/api/game", json={"mode": "online", "players": 2})   # triggers the sweep
    assert server.get_game(st["iid"]) is None
    assert st["code"] not in server.codes


def test_an_online_game_does_not_hoard_an_undo_stack(client):
    st, host = host_game(client, players=2)
    _, a = join(client, st["code"], "Ana")
    iid, tokens = st["iid"], {1: host, 2: a["token"]}
    for _ in range(6):
        state = client.get(f"/api/game/{iid}?token={host}").get_json()["state"]
        mv = state["legal"][0]
        client.post(f"/api/game/{iid}/move",
                    json={"x": mv[0], "y": mv[1], "token": tokens[state["current_player"]]})
    g = server.get_game(iid)
    assert g.state()["move_count"] == 6
    assert g.board.history == [], "online play has no undo, so it should keep no history"
