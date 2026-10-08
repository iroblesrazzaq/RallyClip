import Foundation

public actor Backend {
    public let baseURL: URL
    private let token: String
    private let session: URLSession
    public init(port: Int, token: String) {
        baseURL = URL(string: "http://127.0.0.1:\(port)")!
        self.token = token
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        session = URLSession(configuration: configuration)
    }

    public func request<T: Decodable>(_ path: String, method: String = "GET", body: Data? = nil) async throws -> T {
        let data = try await data(path, method: method, body: body)
        return try JSONDecoder().decode(T.self, from: data)
    }
    private func data(_ path: String, method: String = "GET", body: Data? = nil) async throws -> Data {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method; request.httpBody = body
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw RallyError.message("The analysis service did not respond.") }
        guard (200..<300).contains(response.statusCode) else {
            let message = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
            throw RallyError.message(message ?? "Request failed (\(response.statusCode)).")
        }
        return data
    }

    public func library() async throws -> [Match] {
        struct Response: Decodable { let items: [Match] }
        let response: Response = try await request("api/library")
        return response.items
    }
    public func sources(_ id: String) async throws -> [Source] {
        struct Response: Decodable { let sources: [Source] }
        let response: Response = try await request("api/native/library/\(id)/sources")
        return response.sources
    }
    public func points(_ id: String) async throws -> SegmentResponse { try await request("api/library/\(id)/segments") }
    public func save(_ points: [Point], id: String) async throws -> SegmentResponse {
        struct Body: Encodable { let segments: [Point] }
        return try await request("api/library/\(id)/segments", method: "PUT", body: JSONEncoder().encode(Body(segments: points)))
    }
    public func reset(_ id: String) async throws -> SegmentResponse {
        try await request("api/library/\(id)/segments/edits", method: "DELETE")
    }
    public func importMatch(paths: [String], name: String) async throws -> String {
        struct Body: Encodable { let paths: [String]; let name: String }
        struct Response: Decodable { let job_id: String }
        let response: Response = try await request("api/native/import", method: "POST", body: JSONEncoder().encode(Body(paths: paths, name: name)))
        return response.job_id
    }
    public func job(_ id: String) async throws -> Job { try await request("api/progress/\(id)") }
    public func cancel(_ id: String) async throws {
        let _: [String: String] = try await request("api/cancel/\(id)", method: "POST")
    }
    public func delete(_ id: String) async throws {
        let _: [String: String] = try await request("api/library/\(id)", method: "DELETE")
    }
}
