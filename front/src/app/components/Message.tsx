"use client";
import React, { useState, useEffect } from "react";

const SingleClient = ({ userId }: { userId: number }) => {

  const [message, setMessage] = useState("");
  const [receivedMessage, setReceivedMessage] = useState("");
  const [client, setClient] = useState<WebSocket | null>(null);

  useEffect(() => {
    const newClient = new WebSocket(`ws://192.168.127.253:8080/chat/${userId}`);
    newClient.onopen = () => {
      console.log(`WebSocket Client Connected for user ${userId}`); 
    };
    newClient.onmessage = (message) => {
      const parsemessage = JSON.parse(message.data).message;
      setReceivedMessage(parsemessage);
      console.log(parsemessage);
    };
    setClient(newClient);

    return () => {
      newClient.close();
    };
  }, [userId]);

  const sendMessage = () => {
    if (message.trim() !== "" && client) {
      client.send(JSON.stringify({ message: message }));
      setMessage("");
    }
  };

  return (
    <div>
      <div>{receivedMessage && <p>{receivedMessage}</p>}</div>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
};

const Message = () => {
  return (
    <div>
      <SingleClient userId={1} />
      <SingleClient userId={2} />
      <SingleClient userId={3} />
    </div>
  );
};

export default Message;
