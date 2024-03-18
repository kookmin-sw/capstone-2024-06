/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  env: {
    Localhost: "http://10.223.114.14:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '10.223.114.14', 'k.kakaocdn.net'], 
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
