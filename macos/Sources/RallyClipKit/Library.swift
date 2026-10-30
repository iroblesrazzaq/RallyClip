import Foundation

public enum LibrarySort: String, CaseIterable, Sendable {
    case date, name, custom
    public var label: String {
        switch self {
        case .date: return "Date Processed"
        case .name: return "Name"
        case .custom: return "Custom Order"
        }
    }
}

public enum LibraryOrder {
    /// Date: newest first. Name: A to Z, numbers in numeric order. Custom: the saved
    /// order, with matches it does not hold yet (new imports) first, newest first.
    public static func sorted(_ matches: [Match], by sort: LibrarySort, custom: [String]) -> [Match] {
        let newest = matches.sorted { a, b in
            let (x, y) = (a.created_ts ?? 0, b.created_ts ?? 0)
            if x != y { return x > y }
            if a.created != b.created { return a.created > b.created }
            return a.id > b.id
        }
        switch sort {
        case .date: return newest
        case .name:
            let rank = Dictionary(uniqueKeysWithValues: newest.enumerated().map { ($1.id, $0) })
            return newest.sorted { a, b in
                let order = a.name.localizedStandardCompare(b.name)
                return order == .orderedSame ? rank[a.id]! < rank[b.id]! : order == .orderedAscending
            }
        case .custom:
            let rank = Dictionary(custom.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
            return newest.filter { rank[$0.id] == nil } + newest.filter { rank[$0.id] != nil }.sorted { rank[$0.id]! < rank[$1.id]! }
        }
    }

    /// `ids` after a list drag: the rows at `source` go before the row that was at `destination`.
    public static func move(_ ids: [String], from source: IndexSet, to destination: Int) -> [String] {
        let moving = source.filter { ids.indices.contains($0) }.map { ids[$0] }
        var rest = ids.enumerated().filter { !source.contains($0.offset) }.map(\.element)
        let at = min(rest.count, max(0, destination - source.filter { $0 < destination }.count))
        rest.insert(contentsOf: moving, at: at)
        return rest
    }
}
