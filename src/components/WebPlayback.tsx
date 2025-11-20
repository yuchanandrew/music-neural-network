import React, { useState, useEffect, useRef } from "react";
import muse from "../../server/sentiment_model/muse_v3.json";

declare global {
  interface Window {
    onSpotifyWebPlaybackSDKReady?: () => void;
    Spotify?: any;
  }
}
export {};

interface Props {
  token: string;
}

const WebPlayback = ({ token }: Props) => {
  const playerRef = useRef<any>(null);
  const deviceIdRef = useRef<any>(null);

  const [isPaused, setPaused] = useState(false);
  const [isActive, setActive] = useState(false);
  const [currentTrack, setTrack] = useState<any>(null);

  const [status, setStatus] = useState<number | null>(null);
  const [currentIndex, setCurrentIndex] = useState<number | null>(null);
  const [songInPlayback, setSongInPlayback] = useState<string | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket("ws://localhost:8080");

      ws.onopen = () => {
        console.log("Connected to wss");
      };

      ws.onclose = () => {
        console.log("Disconnected from wss - reconnecting in 1s...");
        reconnectTimeout = setTimeout(connect, 1000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };

      ws.onmessage = (message) => {
        const newData = JSON.parse(message.data);
        setStatus(newData.status);
        setCurrentIndex(newData.index);
        setSongInPlayback(newData.spotify_id);
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }, []);

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://sdk.scdn.co/spotify-player.js";
    script.async = true;
    document.body.appendChild(script);

    window.onSpotifyWebPlaybackSDKReady = () => {
      const player = new window.Spotify.Player({
        name: "Web Playback SDK",
        getOAuthToken: (cb: any) => cb(token),
        volume: 1.0,
      });

      playerRef.current = player;

      player.addListener("ready", async ({ device_id }: any) => {
        console.log("Ready with Device ID", device_id);
        deviceIdRef.current = device_id;

        // Make this device active
        await fetch("https://api.spotify.com/v1/me/player", {
          method: "PUT",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            device_ids: [device_id],
            play: false,
          }),
        });
      });

      player.addListener("not_ready", ({ device_id }: any) => {
        console.log("Device ID offline", device_id);
      });

      player.addListener("player_state_changed", (state: any) => {
        if (!state) return;

        setTrack(state.track_window.current_track);
        setPaused(state.paused);

        player.getCurrentState().then((s: any) => {
          setActive(!!s);
        });
      });

      player.connect();
    };
  }, [token]);

  const player = playerRef.current;

  useEffect(() => {
    const startPlayback = async () => {
      if (!songInPlayback || !player) return;

      try {
        // Start playback
        await fetch(
          `https://api.spotify.com/v1/me/player/play?device_id=${deviceIdRef.current}`,
          {
            method: "PUT",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              uris: [`spotify:track:${songInPlayback}`],
              position_ms: 45000,
            }),
          }
        );
      } catch (error) {
        console.error("Error starting playback:", error);
      }
    };

    startPlayback();
  }, [songInPlayback, player, token]);

  // Get the next song in json
  useEffect(() => {});

  return (
    <div className="container flex-col">
      <div className="main-wrapper flex-col">
        {currentTrack?.album?.images?.[0]?.url && (
          <img
            src={currentTrack.album.images[0].url}
            className="now-playing__cover"
            alt=""
          />
        )}

        <div className="now-playing__side">
          <div className="now-playing__name">{currentTrack?.name}</div>
          <div className="now-playing__artist">
            {currentTrack?.artists?.[0]?.name}
          </div>
        </div>
        <button
          className="btn-spotify"
          onClick={() => player && player.previousTrack()}
        >
          &lt;&lt;
        </button>

        <button
          id="target"
          className="btn-spotify"
          onClick={() =>
            player && player.togglePlay() && console.log("Clicked")
          }
        >
          {isPaused ? "PLAY" : "PAUSE"}
        </button>

        <button
          className="btn-spotify"
          onClick={() => player && player.nextTrack()}
        >
          &gt;&gt;
        </button>
      </div>
      <div>Status: {status}</div>
      <div>Song: {songInPlayback}</div>
    </div>
  );
};

export default WebPlayback;
