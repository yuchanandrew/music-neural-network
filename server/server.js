import express from "express";
import dotenv from "dotenv";
import * as crypto from "node:crypto";
import axios from "axios";
import cors from "cors";

dotenv.config();

const app = express();
app.use(express.json());

// Check if client and secret are correct
// console.log("Client: ", process.env.SPOTIFY_PLAYER_CLIENT);
// console.log("Secret: ", process.env.SPOTIFY_PLAYER_SECRET);

app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

const redirect_uri = "http://127.0.0.1:3000/callback";

app.get("/auth/login", async (req, res) => {
  const state = crypto.randomBytes(16).toString("hex");
  const scope = "streaming user-read-email user-read-private";
  // ("user-read-recently-played user-read-private user-read-email user-read-currently-playing user-read-playback-state");

  const params = new URLSearchParams({
    response_type: "code",
    client_id: process.env.SPOTIFY_PLAYER_CLIENT,
    scope: scope,
    redirect_uri: redirect_uri,
    state: state,
  });

  res.redirect("https://accounts.spotify.com/authorize?" + params.toString());
});

app.get("/callback", async (req, res) => {
  const code = req.query.code || null;
  const state = req.query.state || null;

  const params = new URLSearchParams({
    error: "state_mismatch",
  });

  if (state === null) {
    res.redirect("/#" + params.toString());
  } else {
    const response = await fetch("https://accounts.spotify.com/api/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization:
          "Basic " +
          Buffer.from(
            process.env.SPOTIFY_PLAYER_CLIENT +
              ":" +
              process.env.SPOTIFY_PLAYER_SECRET
          ).toString("base64"),
      },
      body: new URLSearchParams({
        code: code,
        redirect_uri: redirect_uri,
        grant_type: "authorization_code",
      }),
    });

    const data = await response.json();
    console.log("Token data:", data);
  }
});

// Implement this in every request so that it refreshes
const refresh_token = async () => {
  const response = await fetch("https://accounts.spotify.com/api/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization:
        "Basic " +
        Buffer.from(
          process.env.SPOTIFY_PLAYER_CLIENT +
            ":" +
            process.env.SPOTIFY_PLAYER_SECRET
        ).toString("base64"),
    },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: process.env.PLAYER_REFRESH_TOKEN,
    }),
  });

  const data = await response.json();

  return data;
};

console.log("Refresh Token: ", await refresh_token());

app.get("/auth/token", async (req, res) => {
  const response = await refresh_token();

  res.status(200).json({ message: "Retrieved token.", token: response });
});

/**************************************** IMPLEMENTATION ****************************************/

app.listen(3000, () => console.log("Server is running on port 3000."));
