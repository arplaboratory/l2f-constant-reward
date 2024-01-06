#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <boost/beast/websocket.hpp>
#include <filesystem>
#include <fstream>
#include <rl_tools/operations/cpu_mux.h>
#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/ui.h>
namespace rlt = rl_tools;

//#include "../td3/parameters.h"
#include "../training.h"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

class websocket_session;

class State{
public:
    std::vector<int> namespaces;
    std::vector<std::weak_ptr<websocket_session>> ui_sessions;
    std::vector<std::weak_ptr<websocket_session>> backend_sessions;
    std::mutex mutex;
    int request_count = 0;
    std::string now(){
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::string now_s = std::ctime(&now_c);
        now_s.pop_back();
        return now_s;
    }
};

class websocket_session : public std::enable_shared_from_this<websocket_session> {
public:
    enum class TYPE{
        UI,
        BACKEND
    };
private:
    beast::websocket::stream<tcp::socket> ws_;
    std::vector<std::string> queue_;

    State& state;
    TYPE type;
public:
    explicit websocket_session(tcp::socket socket, State& state, TYPE type) : ws_(std::move(socket)), state(state), type(type) {}

    template<class Body>
    void run(http::request<Body>&& req) {
        ws_.async_accept(req, beast::bind_front_handler(&websocket_session::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec){
        if(ec) return;
        if(type == TYPE::UI){
            std::cout << "UI connected" << std::endl;
        }
        else if(type == TYPE::BACKEND){
            int new_namespace;
            state.mutex.lock();
            new_namespace = state.namespaces.size();
            state.namespaces.push_back(new_namespace);
            state.mutex.unlock();
            std::cout << "Backend connected: " << new_namespace << std::endl;
            nlohmann::json message;
            message["channel"] = "handshake";
            message["data"]["namespace"] = std::to_string(new_namespace);
//            ws_.async_write(net::buffer(message.dump()), beast::bind_front_handler(&websocket_session::on_write, shared_from_this()));
            send_message(message.dump());
        }
        state.mutex.lock();
        if(type == websocket_session::TYPE::UI){
            state.ui_sessions.push_back(shared_from_this());
        }
        else{
            state.backend_sessions.push_back(shared_from_this());
        }
        state.mutex.unlock();
        do_read();
    }

    void do_read() {
        ws_.async_read(buffer_, beast::bind_front_handler(&websocket_session::on_read, shared_from_this()));
    }

    void refresh(){
        if(false){
            // terminate connection here
            ws_.async_close(beast::websocket::close_code::normal, beast::bind_front_handler(&websocket_session::on_close, shared_from_this()));
        }
    }

    void on_close(beast::error_code ec) {
        if(ec) {
            std::cerr << "WebSocket close failed: " << ec.message() << std::endl;
            return;
        }
        ws_.next_layer().shutdown(tcp::socket::shutdown_both, ec);
        ws_.next_layer().close(ec);
    }

    void send_message(std::string message){
        boost::asio::post(ws_.get_executor(), beast::bind_front_handler( &websocket_session::on_send_message, shared_from_this(), message));
//        ws_.async_write(net::buffer(message.dump()), beast::bind_front_handler(&websocket_session::on_write, shared_from_this()));
    }
    void on_send_message(std::string message) {
        // Always add to queue
        queue_.push_back(message);

        // Are we already writing?
        if(queue_.size() > 1)
            return;

        // We are not currently writing, so send this immediately
        ws_.async_write(
                net::buffer(queue_.front()),
                beast::bind_front_handler(
                        &websocket_session::on_write,
                        shared_from_this()));
    }


    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if(ec) return;

        auto message_string = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        auto message = nlohmann::json::parse(message_string);

        state.mutex.lock();
        if(type == TYPE::UI){
            for(auto& backend_session : state.backend_sessions){
                if(auto backend_session_ptr = backend_session.lock()){
                    backend_session_ptr->send_message(message.dump());
                }
            }
        }
        else if(type == TYPE::BACKEND){
            for(auto& ui_session : state.ui_sessions){
                if(auto ui_session_ptr = ui_session.lock()){
                    ui_session_ptr->send_message(message.dump());
                }
            }
        }
        state.mutex.unlock();


        if(message["channel"] == "startTraining"){
            std::cout << "startTraining message received" << std::endl;
        }
        do_read();
    }

    void on_write(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        // Handle the error, if any
        if(ec)
            return;

        // Remove the string from the queue
        queue_.erase(queue_.begin());

        // Send the next message if any
        if(! queue_.empty())
            ws_.async_write(
                    net::buffer(queue_.front()),
                    beast::bind_front_handler(
                            &websocket_session::on_write,
                            shared_from_this()));
    }

private:
    beast::flat_buffer buffer_;
};


class http_connection: public std::enable_shared_from_this<http_connection>
{
public:
    http_connection(tcp::socket socket, State& state): socket_(std::move(socket)), state(state){ }
    void start(){
        read_request();
        check_deadline();
    }

private:
    State& state;
    tcp::socket socket_;
    beast::flat_buffer buffer_{8192};
    http::request<http::dynamic_body> request_;
    http::response<http::dynamic_body> response_;
    net::steady_timer deadline_{socket_.get_executor(), std::chrono::seconds(60)};
    void read_request(){
        auto self = shared_from_this();

        http::async_read(
                socket_,
                buffer_,
                request_,
                [self](beast::error_code ec,
                       std::size_t bytes_transferred)
                {
                    boost::ignore_unused(bytes_transferred);
                    if(!ec)
                        self->process_request();
                });
    }

    void process_request(){
        response_.version(request_.version());
        response_.keep_alive(false);

        switch(request_.method())
        {
            case http::verb::get:
                response_.result(http::status::ok);
                response_.set(http::field::server, "Beast");
                create_response();
                break;

            default:
                // We return responses indicating an error if
                // we do not recognize the request method.
                response_.result(http::status::bad_request);
                response_.set(http::field::content_type, "text/plain");
                beast::ostream(response_.body())
                        << "Invalid request-method '"
                        << std::string(request_.method_string())
                        << "'";
                break;
        }
        write_response();
    }

    void create_response(){
        if(request_.target() == "/count"){
            response_.set(http::field::content_type, "text/html");
            beast::ostream(response_.body())
                    << "<html>\n"
                    <<  "<head><title>Request count</title></head>\n"
                    <<  "<body>\n"
                    <<  "<h1>Request count</h1>\n"
                    <<  "<p>There have been "
                    <<  state.request_count
                    <<  " requests so far.</p>\n"
                    <<  "</body>\n"
                    <<  "</html>\n";
        }
        else if(request_.target() == "/time"){
            response_.set(http::field::content_type, "text/html");
            beast::ostream(response_.body())
                    <<  "<html>\n"
                    <<  "<head><title>Current time</title></head>\n"
                    <<  "<body>\n"
                    <<  "<h1>Current time</h1>\n"
                    <<  "<p>The current time is "
                    <<  state.now()
                    <<  " seconds since the epoch.</p>\n"
                    <<  "</body>\n"
                    <<  "</html>\n";
        }
        else if(request_.target() == "/ui"){
            maybe_upgrade(websocket_session::TYPE::UI);
        }
        else if(request_.target() == "/backend"){
            maybe_upgrade(websocket_session::TYPE::BACKEND);
        }
        else{
            std::filesystem::path path(std::string(request_.target()));
            if(path.empty() || path == "/"){
                path = "/index.html";
            }
            path = "src/ui/static" + path.string();
            // check if file at path exists

            if(std::filesystem::exists(path)){
                response_.result(http::status::ok);
                // check extension and use correct content_type
                if(path.extension() == ".html")
                    response_.set(http::field::content_type, "text/html");
                else if(path.extension() == ".js")
                    response_.set(http::field::content_type, "application/javascript");
                else if(path.extension() == ".css")
                    response_.set(http::field::content_type, "text/css");
                else if(path.extension() == ".png")
                    response_.set(http::field::content_type, "image/png");
                else if(path.extension() == ".jpg")
                    response_.set(http::field::content_type, "image/jpeg");
                else if(path.extension() == ".gif")
                    response_.set(http::field::content_type, "image/gif");
                else if(path.extension() == ".ico")
                    response_.set(http::field::content_type, "image/x-icon");
                else if(path.extension() == ".txt")
                    response_.set(http::field::content_type, "text/plain");
                else
                    response_.set(http::field::content_type, "application/octet-stream");
                beast::ostream(response_.body()) << std::ifstream(path).rdbuf();
            }
            else{
                response_.result(http::status::not_found);
                response_.set(http::field::content_type, "text/plain");
                beast::ostream(response_.body()) << "File not found\r\n";
                std::cout << "File not found: " << path << " (you might need to run \"get_dependencies.sh\" to download the UI dependencies into the static folder)" << std::endl;
            }

//            response_.result(http::status::not_found);
//            response_.set(http::field::content_type, "text/plain");
//            beast::ostream(response_.body()) << "File not found\r\n";
        }
    }
    void maybe_upgrade(websocket_session::TYPE type) {
        if (beast::websocket::is_upgrade(request_)) {
            // Construct the WebSocket session and run it
            auto ws_session = std::make_shared<websocket_session>(std::move(socket_), state, type);
            ws_session->run(std::move(request_));
        }
    }


    void write_response(){
        auto self = shared_from_this();

        response_.content_length(response_.body().size());

        http::async_write(
                socket_,
                response_,
                [self](beast::error_code ec, std::size_t)
                {
                    self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                    self->deadline_.cancel();
                });
    }

    void check_deadline(){
        auto self = shared_from_this();

        deadline_.async_wait(
                [self](beast::error_code ec){
                    if(!ec){
                        self->socket_.close(ec);
                    }
                });
    }
};

void http_server(tcp::acceptor& acceptor, tcp::socket& socket, State& state){
    acceptor.async_accept(socket, [&](beast::error_code ec){
        if(!ec)
            std::make_shared<http_connection>(std::move(socket), state)->start();
        http_server(acceptor, socket, state);
    });
}

int main(int argc, char* argv[]) {
    std::cout << "Note: This executable should be executed in the context (working directory) of the main repo e.g. ./build/src/rl_environments_multirotor_ui 0.0.0.0 8000" << std::endl;
    State state;
    try{
        // Check command line arguments.
        if(argc != 3){
            std::cerr << "Usage: " << argv[0] << " <address> <port> (e.g. \'0.0.0.0 8000\' for localhost 8000)\n";
            return EXIT_FAILURE;
        }

        auto const address = net::ip::make_address(argv[1]);
        unsigned short port = static_cast<unsigned short>(std::atoi(argv[2]));

        net::io_context ioc{1};

        tcp::acceptor acceptor{ioc, {address, port}};
        tcp::socket socket{ioc};
        http_server(acceptor, socket, state);

        std::cout << "Web interface coming up at: http://" << address << ":" << port << std::endl;

        ioc.run();
    }
    catch(std::exception const& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}