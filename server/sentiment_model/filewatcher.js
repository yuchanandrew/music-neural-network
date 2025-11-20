import chokidar from "chokidar";
import fs from "fs";
import WebSocket, { WebSocketServer } from "ws";

const watcher = chokidar.watch("./current_index.json");

const wss = new WebSocketServer({ port: 8080 });

wss.on("connection", () => {
  console.log("Client connected");
});

watcher.on("change", (path) => {
  const data = fs.readFileSync("./current_index.json", "utf-8");
  wss.clients.forEach((client) => client.send(data));
});

console.log("WebSocket Server running on port 8080");
