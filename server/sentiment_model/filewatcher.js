import chokidar from "chokidar";
import fs from "fs";
import WebSocket, { WebSocketServer } from "ws";

const watcher = chokidar.watch("./current_index.json");

const wss = new WebSocketServer({ port: 8080 });

wss.on("connection", () => {
  console.log("Client connected");
});

watcher.on("change", (path) => {
  const rawData = fs.readFileSync("./current_index.json", "utf-8");
  const data = JSON.parse(rawData);
  // console.log(data);
  const track = data.track;
  console.log(`Now pointing to ${track}`);
  wss.clients.forEach((client) => client.send(rawData));
});

console.log("WebSocket Server running on port 8080");
