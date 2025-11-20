import React, { useState, useEffect } from "react";
import WebPlayback from "./components/WebPlayback";
import Login from "./components/Login";
import axios from "axios";

const App = () => {
  const [token, setToken] = useState("");

  useEffect(() => {
    const getToken = async () => {
      const response = await axios.get("http://localhost:3000/auth/token");

      // console.log("Reponse: ", response.data.token.access_token);

      setToken(response.data.token.access_token);
    };

    getToken();
  }, []);

  return (
    <div className="flex flex-col w-1/2 h-1/2 justify-center items-center">
      {token === "" ? <Login /> : <WebPlayback token={token} />}
    </div>
  );
};

export default App;
