/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  reactStrictMode: false,
  env: {

    Localhost: "http://10.30.112.18:8080",
    OnlyiP : "10.30.112.18:8080",
  },
  images: {
    domains: ['t1.kakaocdn.net', '175.194.198.155', 'k.kakaocdn.net' , '10.223.115.184'], 

    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
