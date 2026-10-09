import Pymablock.Manuscript.Registry
import Lean.Util.CollectAxioms
import Lean.Util.FoldConsts

/-! Export checked declaration types and dependencies. Report formatting does
not participate in the mathematical proofs. Unknown axioms fail the export. -/
namespace Pymablock.Manuscript
open Lean Meta

private def isProject (n : Name) : Bool :=
  n.toString.startsWith "Pymablock." && !n.toString.startsWith "Pymablock.Manuscript."

private def kind (ci : ConstantInfo) : String :=
  match ci with
  | .thmInfo _ => "theorem"
  | .defnInfo _ => "definition"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"
  | .axiomInfo _ => "axiom"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quotient"

private def dependencies (ci : ConstantInfo) : Array Name := Id.run do
  let mut names := ci.type.getUsedConstants
  if let some value := ci.value? then names := names ++ value.getUsedConstants
  if let .inductInfo info := ci then names := names ++ info.ctors.toArray
  return names.filter isProject

private partial def visit (name : Name) (seen : NameSet) : MetaM NameSet := do
  if seen.contains name then return seen
  let mut seen := seen.insert name
  let ci ← getConstInfo name
  for dep in dependencies ci do seen ← visit dep seen
  return seen

private def declarationJson (name : Name) : MetaM Json := do
  let ci ← getConstInfo name
  let axioms ← collectAxioms name
  for axiomName in axioms do
    unless #[`propext, `Classical.choice, `Quot.sound].contains axiomName do
      throwError "Unapproved axiom {axiomName} in {name}"
  let hypotheses ← forallTelescope ci.type fun xs _ => do
    let mut entries := #[]
    for x in xs do
      let type ← inferType x
      if ← isProp type then
        entries := entries.push <| Json.mkObj [
          ("name", toJson (← x.fvarId!.getUserName).toString),
          ("type", toJson (← ppExpr type).pretty)]
    return entries
  let body ← match ci with
    | .defnInfo info => pure <| toJson (← ppExpr info.value).pretty
    | _ => pure Json.null
  return Json.mkObj [
    ("name", toJson name.toString), ("kind", toJson (kind ci)),
    ("type", toJson (← ppExpr ci.type).pretty), ("body", body),
    ("hypotheses", .arr hypotheses),
    ("dependencies", toJson ((dependencies ci).map Name.toString)),
    ("axioms", toJson (axioms.map Name.toString))]

/-- The ledger includes structure constructors, exposing solver laws and
model definitions instead of hiding them behind a bundled input. -/
def exportCatalog : MetaM Json := do
  let mut seen : NameSet := {}
  for root in roots do seen ← visit root seen
  for occurrence in occurrences do seen ← visit occurrence.declaration seen
  let names := seen.toArray.qsort (fun a b => a.toString < b.toString)
  -- Each declaration gets the same bounded export budget as the catalog grows.
  let declarations ← names.mapM fun name => withCurrHeartbeats (declarationJson name)
  return Json.mkObj [
    ("schema", toJson (2 : Nat)),
    ("roots", toJson (roots.map Name.toString)),
    ("occurrences", .arr (occurrences.map fun row => Json.mkObj [
      ("source", toJson row.source), ("label", toJson row.label), ("declaration", toJson row.declaration.toString),
      ("relation", toJson row.relation)])),
    ("declarations", .arr declarations)]

end Pymablock.Manuscript

set_option maxHeartbeats 1000000 in
run_cmd Lean.Elab.Command.liftTermElabM do
  let catalog ← Pymablock.Manuscript.exportCatalog
  liftM <| IO.FS.createDirAll ".lake/reports"
  liftM <| IO.FS.writeFile ".lake/reports/correspondence.json" catalog.pretty
