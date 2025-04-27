"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { Chart1 } from "@/components/chart-1";
import { Chart2 } from "@/components/chart-2";
import { Particles } from "@/components/magicui/particles";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <section className="w-full flex justify-center py-12 md:py-24 lg:py-32 xl:py-48 relative overflow-hidden">
          <Particles
            className="absolute inset-0 z-0"
            quantity={400}
            ease={80}
            color="#8b5cf6"
            refresh
          />

          <div className="container px-4 md:px-6 relative z-10">
            <motion.div
              className="flex flex-col items-center space-y-6 text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <motion.div
                className="relative"
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <span className="absolute -inset-1 rounded-lg bg-gradient-to-r from-purple-600 to-blue-600 opacity-75 blur-2xl"></span>
                <h1 className="relative text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-blue-500">
                  Detect Deepfakes with Confidence
                </h1>
              </motion.div>

              <motion.p
                className="mx-auto max-w-[700px] text-muted-foreground md:text-xl"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
              >
                Our AI-powered tools help you identify manipulated images,
                videos, and audio with industry-leading accuracy.
              </motion.p>

              <motion.div
                className="flex flex-wrap justify-center gap-4"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                <Button
                  asChild
                  size="lg"
                  className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-purple-500/25"
                >
                  <Link href="/deepfake" className="group">
                    Get Started
                    <motion.span
                      initial={{ x: 0 }}
                      whileHover={{ x: 5 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </motion.span>
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  asChild
                  className="border-purple-300 hover:bg-purple-100/10 hover:text-purple-600 transition-all duration-300"
                >
                  <Link href="#about">Learn More</Link>
                </Button>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Tabs Section */}
        <section className="w-full flex justify-center py-12 md:py-12 lg:py-12">
          <div className="container px-4 md:px-6">
            <h2 className="mb-8 text-center text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-blue-500">
              Deepfake Impact Analysis
            </h2>
            <Tabs defaultValue="timeline" className="mx-auto max-w-4xl">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="timeline">Impact Over Time</TabsTrigger>
                <TabsTrigger value="sectors">Affected Sectors</TabsTrigger>
                <TabsTrigger value="incidents">Notable Incidents</TabsTrigger>
              </TabsList>
              <TabsContent value="timeline" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="w-full">{<Chart2 />}</div>
                </motion.div>
              </TabsContent>
              <TabsContent value="sectors" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="w-full">{<Chart1 />}</div>
                </motion.div>
              </TabsContent>
              <TabsContent value="incidents" className="p-4">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h3 className="mb-4 text-xl font-bold">Notable Incidents</h3>
                  <Table>
                    <TableCaption>
                      Recent significant deepfake incidents
                    </TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Impact</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell>Jan 2023</TableCell>
                        <TableCell>Political</TableCell>
                        <TableCell>Election misinformation campaign</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Severe</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Mar 2023</TableCell>
                        <TableCell>Financial</TableCell>
                        <TableCell>CEO voice fraud ($3.2M loss)</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Severe</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Jun 2023</TableCell>
                        <TableCell>Entertainment</TableCell>
                        <TableCell>Celebrity deepfake controversy</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Moderate</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Sep 2023</TableCell>
                        <TableCell>Educational</TableCell>
                        <TableCell>Academic credential fraud</TableCell>
                        <TableCell>
                          <Badge variant="destructive">Moderate</Badge>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Dec 2023</TableCell>
                        <TableCell>Social Media</TableCell>
                        <TableCell>Viral deepfake news story</TableCell>
                        <TableCell>
                          <Badge variant="outline">Resolved</Badge>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </motion.div>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* FAQ Section */}
        <section
          id="faq"
          className="w-full flex justify-center py-12 md:py-24 lg:py-32"
        >
          <div className="container px-4 md:px-6">
            <h2 className="mb-8 text-center text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-blue-500">
              Frequently Asked Questions
            </h2>
            <div className="mx-auto max-w-3xl">
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger className="text-xl">
                    How accurate is deepfake detection?
                  </AccordionTrigger>
                  <AccordionContent>
                    Our detection tools achieve 94-97% accuracy on benchmark
                    datasets. However, as deepfake technology evolves, we
                    continuously update our models to maintain high detection
                    rates.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                  <AccordionTrigger className="text-xl">
                    Can I detect deepfakes on my smartphone?
                  </AccordionTrigger>
                  <AccordionContent>
                    Yes, our web application is fully responsive and works on
                    smartphones. For optimal performance with large video files,
                    we recommend using a desktop computer.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-3">
                  <AccordionTrigger className="text-xl">
                    How do I report a deepfake?
                  </AccordionTrigger>
                  <AccordionContent>
                    If you've identified a deepfake using our tools, you can
                    report it to the platform where it was shared. Most social
                    media platforms have policies against manipulated media and
                    provide reporting mechanisms.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-4">
                  <AccordionTrigger className="text-xl">
                    Is my uploaded content secure?
                  </AccordionTrigger>
                  <AccordionContent>
                    We prioritize your privacy. All uploaded content is
                    processed securely, not stored permanently, and never shared
                    with third parties. Our analysis happens on encrypted
                    connections and files are deleted after processing.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-5">
                  <AccordionTrigger className="text-xl">
                    What types of deepfakes can be detected?
                  </AccordionTrigger>
                  <AccordionContent>
                    Our tools can detect face-swapped images and videos,
                    synthetic voices, and AI-generated content. We continuously
                    expand our capabilities to address new types of manipulated
                    media as they emerge.
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
