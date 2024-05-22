/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  reactStrictMode: false,
  env: {

    Localhost: `http://192.168.66.253:8080`,
    OnlyiP : "http://192.168.66.253:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '192.168.66.253', 'k.kakaocdn.net' , '10.223.115.184'], 

    remotePatterns: [
      {
        protocol: "https",
        hostname: "**", 
      },
    ],
  },
};

module.exports = nextConfig;
